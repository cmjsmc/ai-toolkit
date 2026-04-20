import fs from 'fs';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { getDecryptedJson, encryptedJsonResponse } from '@/utils/serverEncryption';

export async function POST(request: Request) {
  try {
    const body = await getDecryptedJson(request);
    const { imgPath } = body;
    let datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    if (!imgPath.startsWith(datasetsPath) && !imgPath.startsWith(trainingPath)) {
      return encryptedJsonResponse({ error: 'Invalid image path' }, { status: 400 });
    }

    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$/i.test(imgPath.toLowerCase())) {
      return encryptedJsonResponse({ error: 'Not an image' }, { status: 400 });
    }

    if (!fs.existsSync(imgPath)) {
      return encryptedJsonResponse({ success: true });
    }

    fs.unlinkSync(imgPath);

    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(captionPath)) {
      fs.unlinkSync(captionPath);
    }

    return encryptedJsonResponse({ success: true });
  } catch (error) {
    return encryptedJsonResponse({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
