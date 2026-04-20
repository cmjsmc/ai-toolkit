import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { getDecryptedJson, encryptedJsonResponse } from '@/utils/serverEncryption';

export async function POST(request: Request) {
  try {
    const body = await getDecryptedJson(request);
    const { name } = body;
    let datasetsPath = await getDatasetsRoot();
    let datasetPath = path.join(datasetsPath, name);

    // if folder doesnt exist, ignore
    if (!fs.existsSync(datasetPath)) {
      return encryptedJsonResponse({ success: true });
    }

    // delete it and return success
    fs.rmSync(datasetPath, { recursive: true, force: true });
    return encryptedJsonResponse({ success: true });
  } catch (error) {
    return encryptedJsonResponse({ error: 'Failed to delete dataset' }, { status: 500 });
  }
}
