import os
import weakref
from _weakref import ReferenceType
from typing import TYPE_CHECKING, List, Union
import cv2
import torch
import random
import io

from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms

from toolkit import image_utils
from toolkit.basic import get_quick_signature_string
from toolkit.dataloader_mixins import CaptionProcessingDTOMixin, ImageProcessingDTOMixin, LatentCachingFileItemDTOMixin, \
    ControlFileItemDTOMixin, ArgBreakMixin, PoiFileItemDTOMixin, MaskFileItemDTOMixin, AugmentationFileItemDTOMixin, \
    UnconditionalFileItemDTOMixin, ClipImageFileItemDTOMixin, InpaintControlFileItemDTOMixin, TextEmbeddingFileItemDTOMixin, \
    clean_caption
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds

if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig
    from toolkit.stable_diffusion_model import StableDiffusion

printed_messages = []


def print_once(msg):
    pass


class FileItemDTO(
    LatentCachingFileItemDTOMixin,
    TextEmbeddingFileItemDTOMixin,
    CaptionProcessingDTOMixin,
    ImageProcessingDTOMixin,
    ControlFileItemDTOMixin,
    InpaintControlFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    MaskFileItemDTOMixin,
    AugmentationFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    PoiFileItemDTOMixin,
    ArgBreakMixin,
):
    def __init__(self, *args, **kwargs):
        self.path = kwargs.get('path', '')
        self.in_memory_data = kwargs.get('in_memory_data', None)
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.is_video = self.dataset_config.num_frames > 1
        size_database = kwargs.get('size_database', {})
        dataset_root =  kwargs.get('dataset_root', None)
        self.encode_control_in_text_embeddings = kwargs.get('encode_control_in_text_embeddings', False)
        
        if dataset_root and dataset_root != "":
            # remove dataset root from path
            file_key = self.path.replace(dataset_root, '')
        else:
            file_key = self.path # usually strict path in tar
        
        use_db_entry = False
        if self.in_memory_data is None:
            # Only use signature for disk files
            file_signature = get_quick_signature_string(self.path)
            if file_signature is None:
                raise Exception(f"Error: Could not get file signature for {self.path}")
            
            if file_key in size_database:
                db_entry = size_database[file_key]
                if db_entry is not None and len(db_entry) >= 3 and db_entry[2] == file_signature:
                    use_db_entry = True
        else:
            file_signature = "in_memory"

        if use_db_entry:
            w, h, _ = size_database[file_key]
        elif self.in_memory_data is not None and self.path in self.in_memory_data:
            # Load from memory
            img_bytes = io.BytesIO(self.in_memory_data[self.path])
            try:
                img = exif_transpose(Image.open(img_bytes))
                w, h = img.size
            except Exception as e:
                # If cannot read, set generic size, will fail later probably
                w, h = 1024, 1024
        elif self.is_video:
            # Open the video file
            video = cv2.VideoCapture(self.path)
            if not video.isOpened():
                raise Exception(f"Error: Could not open video file {self.path}")
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w, h = width, height
            video.release()
            if self.in_memory_data is None:
                size_database[file_key] = (width, height, file_signature)
        else:
            if self.dataset_config.fast_image_size:
                try:
                    w, h = image_utils.get_image_size(self.path)
                except image_utils.UnknownImageFormat:
                    img = exif_transpose(Image.open(self.path))
                    w, h = img.size
            else:
                img = exif_transpose(Image.open(self.path))
                w, h = img.size
            if self.in_memory_data is None:
                size_database[file_key] = (w, h, file_signature)
        self.width: int = w
        self.height: int = h
        self.dataloader_transforms = kwargs.get('dataloader_transforms', None)
        super().__init__(*args, **kwargs)

        self.raw_caption: str = kwargs.get('raw_caption', None)
        # we scale first, then crop
        self.scale_to_width: int = kwargs.get('scale_to_width', int(self.width * self.dataset_config.scale))
        self.scale_to_height: int = kwargs.get('scale_to_height', int(self.height * self.dataset_config.scale))
        # crop values are from scaled size
        self.crop_x: int = kwargs.get('crop_x', 0)
        self.crop_y: int = kwargs.get('crop_y', 0)
        self.crop_width: int = kwargs.get('crop_width', self.scale_to_width)
        self.crop_height: int = kwargs.get('crop_height', self.scale_to_height)
        self.flip_x: bool = kwargs.get('flip_x', False)
        self.flip_y: bool = kwargs.get('flip_x', False)
        self.augments: List[str] = self.dataset_config.augments
        self.loss_multiplier: float = self.dataset_config.loss_multiplier

        self.network_weight: float = self.dataset_config.network_weight
        self.is_reg = self.dataset_config.is_reg
        self.prior_reg = self.dataset_config.prior_reg
        self.tensor: Union[torch.Tensor, None] = None

    # Overridden to support in-memory loading
    def load_caption(self: 'FileItemDTO', caption_dict: Union[dict, None]=None):
        if self.raw_caption is not None:
            # we already loaded it
            pass
        elif caption_dict is not None and self.path in caption_dict and "caption" in caption_dict[self.path]:
            self.raw_caption = caption_dict[self.path]["caption"]
            if 'caption_short' in caption_dict[self.path]:
                self.raw_caption_short = caption_dict[self.path]["caption_short"]
                if self.dataset_config.use_short_captions:
                    self.raw_caption = caption_dict[self.path]["caption_short"]
        else:
            # see if prompt file exists
            path_no_ext = os.path.splitext(self.path)[0]
            prompt_ext = self.dataset_config.caption_ext
            prompt_path = path_no_ext + prompt_ext
            short_caption = None
            prompt = ''

            if self.in_memory_data is not None and prompt_path in self.in_memory_data:
                # Load caption from memory
                content = self.in_memory_data[prompt_path]
                if isinstance(content, bytes):
                    prompt = content.decode('utf-8')
                else:
                    prompt = str(content)
                
                # Check json
                if prompt_path.endswith('.json'):
                    prompt = prompt.replace('\r\n', ' ')
                    prompt = prompt.replace('\n', ' ')
                    prompt = prompt.replace('\r', ' ')
                    
                    try:
                        prompt_json = json.loads(prompt)
                        if 'caption' in prompt_json:
                            prompt = prompt_json['caption']
                        if 'caption_short' in prompt_json:
                            short_caption = prompt_json['caption_short']
                            if self.dataset_config.use_short_captions:
                                prompt = short_caption
                        if 'extra_values' in prompt_json:
                            self.extra_values = prompt_json['extra_values']
                    except:
                        pass
                
                prompt = clean_caption(prompt)
                if short_caption is not None:
                    short_caption = clean_caption(short_caption)
                
                if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption

            elif os.path.exists(prompt_path):
                # Fallback to disk
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                    short_caption = None
                    if prompt_path.endswith('.json'):
                        # replace any line endings with commas for \n \r \r\n
                        prompt = prompt.replace('\r\n', ' ')
                        prompt = prompt.replace('\n', ' ')
                        prompt = prompt.replace('\r', ' ')

                        prompt_json = json.loads(prompt)
                        if 'caption' in prompt_json:
                            prompt = prompt_json['caption']
                        if 'caption_short' in prompt_json:
                            short_caption = prompt_json['caption_short']
                            if self.dataset_config.use_short_captions:
                                prompt = short_caption
                        if 'extra_values' in prompt_json:
                            self.extra_values = prompt_json['extra_values']

                    prompt = clean_caption(prompt)
                    if short_caption is not None:
                        short_caption = clean_caption(short_caption)
                    
                    if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption
            else:
                prompt = ''
                if self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption

            if short_caption is None:
                short_caption = self.dataset_config.default_caption
            self.raw_caption = prompt
            self.raw_caption_short = short_caption

        self.caption = self.get_caption()
        if self.raw_caption_short is not None:
            self.caption_short = self.get_caption(short_caption=True)

    # Overridden to support in-memory loading
    def load_and_process_image(
            self,
            transform: Union[None, transforms.Compose],
            only_load_latents=False
    ):
        if self.dataset_config.num_frames > 1:
            self.load_and_process_video(transform, only_load_latents)
            return
        
        if self.is_text_embedding_cached:
            self.load_prompt_embedding()
            
        if self.is_latent_cached:
            self.get_latent()
            if self.has_control_image:
                self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()
            return

        img = None
        try:
            if self.in_memory_data is not None and self.path in self.in_memory_data:
                # Load from memory
                img_bytes = io.BytesIO(self.in_memory_data[self.path])
                img = Image.open(img_bytes)
                img.load() 
            else:
                # Load from disk
                img = Image.open(self.path)
                img.load()
            
            img = exif_transpose(img)
        except Exception as e:
            # Silence error
            pass
        
        if img is None:
            # Return dummy image
            img = Image.new('RGB', (self.width, self.height), color=(127, 127, 127))

        if self.use_alpha_as_mask:
            np_img = np.array(img)
            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        w, h = img.size
        # ... sizing checks ...

        if self.flip_x:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            img = img.resize(
                (int(img.size[0] * self.dataset_config.scale), int(img.size[1] * self.dataset_config.scale)),
                Image.BICUBIC)
            min_img_size = min(img.size)
            if self.dataset_config.random_crop:
                if self.dataset_config.random_scale and min_img_size > self.dataset_config.resolution:
                    scale_size = random.randint(self.dataset_config.resolution, int(min_img_size))
                    scaler = scale_size / min_img_size
                    scale_width = int((img.width + 5) * scaler)
                    scale_height = int((img.height + 5) * scaler)
                    img = img.resize((scale_width, scale_height), Image.BICUBIC)
                img = transforms.RandomCrop(self.dataset_config.resolution)(img)
            else:
                img = transforms.CenterCrop(min_img_size)(img)
                img = img.resize((self.dataset_config.resolution, self.dataset_config.resolution), Image.BICUBIC)

        if self.augments is not None and len(self.augments) > 0:
            for augment in self.augments:
                pass 

        if self.has_augmentations:
            img = self.augment_image(img, transform=transform)
        elif transform:
            img = transform(img)

        self.tensor = img
        if not only_load_latents:
            if self.has_control_image:
                self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()

    def cleanup(self):
        self.tensor = None
        self.cleanup_latent()
        self.cleanup_text_embedding()
        self.cleanup_control()
        self.cleanup_inpaint()
        self.cleanup_clip_image()
        self.cleanup_mask()
        self.cleanup_unconditional()


class DataLoaderBatchDTO:
    def __init__(self, **kwargs):
        try:
            self.file_items: List['FileItemDTO'] = kwargs.get('file_items', None)
            is_latents_cached = self.file_items[0].is_latent_cached
            is_text_embedding_cached = self.file_items[0].is_text_embedding_cached
            self.tensor: Union[torch.Tensor, None] = None
            self.latents: Union[torch.Tensor, None] = None
            self.control_tensor: Union[torch.Tensor, None] = None
            self.control_tensor_list: Union[List[List[torch.Tensor]], None] = None
            self.clip_image_tensor: Union[torch.Tensor, None] = None
            self.mask_tensor: Union[torch.Tensor, None] = None
            self.unaugmented_tensor: Union[torch.Tensor, None] = None
            self.unconditional_tensor: Union[torch.Tensor, None] = None
            self.unconditional_latents: Union[torch.Tensor, None] = None
            self.clip_image_embeds: Union[List[dict], None] = None
            self.clip_image_embeds_unconditional: Union[List[dict], None] = None
            self.sigmas: Union[torch.Tensor, None] = None 
            self.extra_values: Union[torch.Tensor, None] = torch.tensor([x.extra_values for x in self.file_items]) if len(self.file_items[0].extra_values) > 0 else None
            if not is_latents_cached:
                self.tensor: torch.Tensor = torch.cat([x.tensor.unsqueeze(0) for x in self.file_items])
            self.latents: Union[torch.Tensor, None] = None
            if is_latents_cached:
                self.latents = torch.cat([x.get_latent().unsqueeze(0) for x in self.file_items])
            self.prompt_embeds: Union[PromptEmbeds, None] = None
            
            if any([x.control_tensor is not None for x in self.file_items]):
                base_control_tensor = None
                for x in self.file_items:
                    if x.control_tensor is not None:
                        base_control_tensor = x.control_tensor
                        break
                control_tensors = []
                for x in self.file_items:
                    if x.control_tensor is None:
                        control_tensors.append(torch.zeros_like(base_control_tensor))
                    else:
                        control_tensors.append(x.control_tensor)
                self.control_tensor = torch.cat([x.unsqueeze(0) for x in control_tensors])
            
            if any([x.control_tensor_list is not None for x in self.file_items]):
                self.control_tensor_list = []
                for x in self.file_items:
                    if x.control_tensor_list is not None:
                        self.control_tensor_list.append(x.control_tensor_list)
                    else:
                        raise Exception(f"Could not find control tensors for all file items, missing for {x.path}")
                    
            self.inpaint_tensor: Union[torch.Tensor, None] = None
            if any([x.inpaint_tensor is not None for x in self.file_items]):
                base_inpaint_tensor = None
                for x in self.file_items:
                    if x.inpaint_tensor is not None:
                        base_inpaint_tensor = x.inpaint_tensor
                        break
                inpaint_tensors = []
                for x in self.file_items:
                    if x.inpaint_tensor is None:
                        inpaint_tensors.append(torch.zeros_like(base_inpaint_tensor))
                    else:
                        inpaint_tensors.append(x.inpaint_tensor)
                self.inpaint_tensor = torch.cat([x.unsqueeze(0) for x in inpaint_tensors])

            self.loss_multiplier_list: List[float] = [x.loss_multiplier for x in self.file_items]

            if any([x.clip_image_tensor is not None for x in self.file_items]):
                base_clip_image_tensor = None
                for x in self.file_items:
                    if x.clip_image_tensor is not None:
                        base_clip_image_tensor = x.clip_image_tensor
                        break
                clip_image_tensors = []
                for x in self.file_items:
                    if x.clip_image_tensor is None:
                        clip_image_tensors.append(torch.zeros_like(base_clip_image_tensor))
                    else:
                        clip_image_tensors.append(x.clip_image_tensor)
                self.clip_image_tensor = torch.cat([x.unsqueeze(0) for x in clip_image_tensors])

            if any([x.mask_tensor is not None for x in self.file_items]):
                base_mask_tensor = None
                for x in self.file_items:
                    if x.mask_tensor is not None:
                        base_mask_tensor = x.mask_tensor
                        break
                mask_tensors = []
                for x in self.file_items:
                    if x.mask_tensor is None:
                        mask_tensors.append(torch.zeros_like(base_mask_tensor))
                    else:
                        mask_tensors.append(x.mask_tensor)
                self.mask_tensor = torch.cat([x.unsqueeze(0) for x in mask_tensors])

            if any([x.unaugmented_tensor is not None for x in self.file_items]):
                base_unaugmented_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unaugmented_tensor = x.unaugmented_tensor
                        break
                unaugmented_tensor = []
                for x in self.file_items:
                    if x.unaugmented_tensor is None:
                        unaugmented_tensor.append(torch.zeros_like(base_unaugmented_tensor))
                    else:
                        unaugmented_tensor.append(x.unaugmented_tensor)
                self.unaugmented_tensor = torch.cat([x.unsqueeze(0) for x in unaugmented_tensor])

            if any([x.unconditional_tensor is not None for x in self.file_items]):
                base_unconditional_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unconditional_tensor = x.unconditional_tensor
                        break
                unconditional_tensor = []
                for x in self.file_items:
                    if x.unconditional_tensor is None:
                        unconditional_tensor.append(torch.zeros_like(base_unconditional_tensor))
                    else:
                        unconditional_tensor.append(x.unconditional_tensor)
                self.unconditional_tensor = torch.cat([x.unsqueeze(0) for x in unconditional_tensor])

            if any([x.clip_image_embeds is not None for x in self.file_items]):
                self.clip_image_embeds = []
                for x in self.file_items:
                    if x.clip_image_embeds is not None:
                        self.clip_image_embeds.append(x.clip_image_embeds)
                    else:
                        raise Exception("clip_image_embeds is None for some file items")

            if any([x.clip_image_embeds_unconditional is not None for x in self.file_items]):
                self.clip_image_embeds_unconditional = []
                for x in self.file_items:
                    if x.clip_image_embeds_unconditional is not None:
                        self.clip_image_embeds_unconditional.append(x.clip_image_embeds_unconditional)
                    else:
                        raise Exception("clip_image_embeds_unconditional is None for some file items")
            
            if any([x.prompt_embeds is not None for x in self.file_items]):
                base_prompt_embeds = None
                for x in self.file_items:
                    if x.prompt_embeds is not None:
                        base_prompt_embeds = x.prompt_embeds
                        break
                prompt_embeds_list = []
                for x in self.file_items:
                    if x.prompt_embeds is None:
                        prompt_embeds_list.append(base_prompt_embeds)
                    else:
                        prompt_embeds_list.append(x.prompt_embeds)
                self.prompt_embeds = concat_prompt_embeds(prompt_embeds_list)
                    

        except Exception as e:
            raise e

    def get_is_reg_list(self):
        return [x.is_reg for x in self.file_items]

    def get_network_weight_list(self):
        return [x.network_weight for x in self.file_items]

    def get_caption_list(
            self,
            trigger=None,
            to_replace_list=None,
            add_if_not_present=True
    ):
        return [x.caption for x in self.file_items]

    def get_caption_short_list(
            self,
            trigger=None,
            to_replace_list=None,
            add_if_not_present=True
    ):
        return [x.caption_short for x in self.file_items]

    def cleanup(self):
        del self.latents
        del self.tensor
        del self.control_tensor
        for file_item in self.file_items:
            file_item.cleanup()
    
    @property
    def dataset_config(self) -> 'DatasetConfig':
        if len(self.file_items) > 0:
            return self.file_items[0].dataset_config
        else:
            return None
