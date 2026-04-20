import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';
import { encryptedJsonResponse } from '@/utils/serverEncryption';

export async function GET() {
  try {
    let datasetsPath = await getDatasetsRoot();

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetsPath)) {
      fs.mkdirSync(datasetsPath);
    }

    // find all the folders in the datasets folder
    let folders = fs
      .readdirSync(datasetsPath, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .filter(dirent => !dirent.name.startsWith('.'))
      .map(dirent => dirent.name);

    return encryptedJsonResponse(folders);
  } catch (error) {
    return encryptedJsonResponse({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
