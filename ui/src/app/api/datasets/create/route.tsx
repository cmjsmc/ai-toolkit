import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { getDecryptedJson, encryptedJsonResponse } from '@/utils/serverEncryption';

export async function POST(request: Request) {
  try {
    const body = await getDecryptedJson(request);
    let { name } = body;
    // clean name by making lower case,  removing special characters, and replacing spaces with underscores
    name = name.toLowerCase().replace(/[^a-z0-9]+/g, '_');

    let datasetsPath = await getDatasetsRoot();
    let datasetPath = path.join(datasetsPath, name);

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetPath)) {
      fs.mkdirSync(datasetPath);
    }

    return encryptedJsonResponse({ success: true, name: name });
  } catch (error) {
    return encryptedJsonResponse({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
