// src/utils/serverEncryption.ts
import { NextRequest, NextResponse } from 'next/server';
import { encryptPayload, decryptPayload } from './crypto';

export async function getDecryptedJson(request: NextRequest | Request) {
  let body;
  try {
    body = await request.json();
  } catch (e) {
    return null;
  }

  if (body && typeof body === 'object' && 'encryptedPayload' in body) {
    const password = process.env.AI_TOOLKIT_AUTH || '';
    if (password) {
      try {
        return await decryptPayload(body.encryptedPayload, password);
      } catch (err) {
        console.error('Failed to decrypt request body', err);
        return body;
      }
    }
  }
  return body;
}

export async function encryptedJsonResponse(data: any, init?: ResponseInit) {
  const password = process.env.AI_TOOLKIT_AUTH || '';
  if (password) {
    const encrypted = await encryptPayload(data, password);
    return NextResponse.json(encrypted, init);
  }
  return NextResponse.json(data, init);
}
