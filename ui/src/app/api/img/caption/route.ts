import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';
import { getDecryptedJson, encryptedJsonResponse } from '@/utils/serverEncryption';

export async function POST(request: Request) {
  try {
    const body = await getDecryptedJson(request);
    const { imgPath, caption } = body;
    let datasetsPath = await getDatasetsRoot();
    
    if (!imgPath.startsWith(datasetsPath)) {
      return encryptedJsonResponse({ error: 'Invalid image path' }, { status: 400 });
    }

    if (!fs.existsSync(imgPath)) {
      return encryptedJsonResponse({ error: 'Image does not exist' }, { status: 404 });
    }

    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    fs.writeFileSync(captionPath, caption);

    return encryptedJsonResponse({ success: true });
  } catch (error) {
    return encryptedJsonResponse({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
