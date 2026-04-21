from .flux2_model import Flux2Model
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    def load_te(self):
        import hashlib
        from toolkit.paths import MODELS_PATH

        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        
        dtype = self.torch_dtype
        
        # --- CACHE SETUP ---
        cache_dir = os.path.join(MODELS_PATH, "quantized_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        te_hash_str = f"{self.flux2_klein_te_path}_{self.model_config.qtype_te}"
        te_hash = hashlib.md5(te_hash_str.encode()).hexdigest()
        te_cache_path = os.path.join(cache_dir, f"te_{te_hash}.pt")
        # -------------------

        if self.model_config.quantize_te and os.path.exists(te_cache_path):
            self.print_and_status_update("Loading CACHED quantized Qwen3")
            
            # Create shell on meta device to save RAM
            with torch.device("meta"):
                text_encoder = Qwen3ForCausalLM.from_pretrained(
                    self.flux2_klein_te_path, torch_dtype=dtype
                )
                
            # Prepare layers for quantization
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            
            # Load quantized tensor subclasses
            cached_sd = torch.load(te_cache_path, map_location="cpu", weights_only=False)
            text_encoder.load_state_dict(cached_sd, assign=True)
            del cached_sd
        else:
            self.print_and_status_update("Loading Qwen3")
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                self.flux2_klein_te_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            
            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing Qwen3")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                
                self.print_and_status_update("Saving quantized Qwen3 to cache...")
                torch.save(text_encoder.state_dict(), te_cache_path)

        if not self.model_config.low_vram and not self.model_config.quantize_te:
            text_encoder.to(self.device_torch, dtype=dtype)
            
        flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(self.flux2_klein_te_path)
        return text_encoder, tokenizer


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
