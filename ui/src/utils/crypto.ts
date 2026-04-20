// src/utils/crypto.ts

// Helper to derive a 256-bit AES-GCM key from the password
async function getKey(password: string): Promise<CryptoKey> {
  const enc = new TextEncoder();
  const hash = await crypto.subtle.digest('SHA-256', enc.encode(password));
  return crypto.subtle.importKey('raw', hash, { name: 'AES-GCM' }, false, ['encrypt', 'decrypt']);
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(buffer).toString('base64');
  }
  let binary = '';
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(base64, 'base64');
    return new Uint8Array(buf).buffer;
  }
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

export async function encryptPayload(data: any, password?: string) {
  if (!password) return data;
  try {
    const key = await getKey(password);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encodedData = new TextEncoder().encode(JSON.stringify(data));

    const ciphertext = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, encodedData);

    const combined = new Uint8Array(12 + ciphertext.byteLength);
    combined.set(iv, 0);
    combined.set(new Uint8Array(ciphertext), 12);

    return { encryptedPayload: arrayBufferToBase64(combined.buffer) };
  } catch (e) {
    console.error('Encryption failed', e);
    return data; // Fallback to plaintext if crypto fails
  }
}

export async function decryptPayload(encryptedBase64: string, password?: string) {
  if (!password || !encryptedBase64) return encryptedBase64;
  try {
    const key = await getKey(password);
    const combined = new Uint8Array(base64ToArrayBuffer(encryptedBase64));

    const iv = combined.slice(0, 12);
    const ciphertext = combined.slice(12);

    const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext);
    const decodedStr = new TextDecoder().decode(decrypted);
    return JSON.parse(decodedStr);
  } catch (e) {
    console.error('Decryption failed', e);
    throw e;
  }
}
