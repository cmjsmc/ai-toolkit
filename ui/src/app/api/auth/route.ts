import { encryptedJsonResponse } from '@/utils/serverEncryption';

export async function GET() {
  // if this gets hit, auth has already been verified
  return encryptedJsonResponse({ isAuthenticated: true });
}
