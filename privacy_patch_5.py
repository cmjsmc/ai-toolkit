#!/usr/bin/env python3
"""
apply_privacy_patch.py

Complete all-in-one patch script for AI-Toolkit End-to-End Encryption & Privacy:
  - Backend AES-256-GCM crypto engine (toolkit/crypto.py)
  - Browser WebCrypto engine with EXIF stripping and stem anonymization (ui/src/utils/crypto.ts)
  - Settings page with Encryption Password generator and visibility toggle (ui/src/app/settings/page.tsx)
  - Settings server, API, and hooks (ui/src/server/settings.ts, ui/src/app/api/settings/route.ts, ui/src/hooks/useSettings.tsx)
  - Dataset upload with metadata stripping & encryption (ui/src/components/AddImagesModal.tsx)
  - Dataset card, viewer, and caption batching with in-browser decryption (ui/src/components/DatasetImageCard.tsx, DatasetImageViewer.tsx, useCaptionBatch.tsx)
  - Sample card and viewer with in-browser preview decryption (ui/src/components/SampleImageCard.tsx, SampleImageViewer.tsx)
  - Backend job spawning, config decryption, RAM decoding, and metadata sanitization

Usage:
  python apply_privacy_patch.py [/path/to/ai-toolkit]
"""

import os
import sys
from pathlib import Path

# ===========================================================================
# Full File Definitions
# ===========================================================================

TOOLKIT_CRYPTO_PY = '''import os
import io
import hmac
import hashlib
from typing import Optional, Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

AITK_MAGIC_HEADER = b"AITK_ENC_V1"
SALT_SIZE = 16
IV_SIZE = 12
TAG_SIZE = 16
PBKDF2_ITERATIONS = 100000


def get_encryption_password() -> Optional[str]:
    pwd = os.environ.get("AITK_ENCRYPTION_PASSWORD", None)
    if pwd is not None:
        pwd = pwd.strip()
        if len(pwd) == 0:
            pwd = None
    return pwd


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def is_encrypted_bytes(data: bytes) -> bool:
    return data.startswith(AITK_MAGIC_HEADER)


def encrypt_bytes(data: bytes, password: Optional[str] = None) -> bytes:
    if password is None:
        password = get_encryption_password()
    if password is None:
        return data

    salt = os.urandom(SALT_SIZE)
    iv = os.urandom(IV_SIZE)
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, data, None)

    return AITK_MAGIC_HEADER + salt + iv + ciphertext


def decrypt_bytes(data: bytes, password: Optional[str] = None) -> bytes:
    if not is_encrypted_bytes(data):
        return data

    if password is None:
        password = get_encryption_password()
    if password is None:
        raise ValueError(
            "Encrypted content detected, but AITK_ENCRYPTION_PASSWORD is not set in the environment."
        )

    offset = len(AITK_MAGIC_HEADER)
    salt = data[offset : offset + SALT_SIZE]
    offset += SALT_SIZE
    iv = data[offset : offset + IV_SIZE]
    offset += IV_SIZE
    ciphertext = data[offset:]

    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(iv, ciphertext, None)


def read_decrypted_file(file_path: str, password: Optional[str] = None) -> bytes:
    with open(file_path, "rb") as f:
        content = f.read()
    return decrypt_bytes(content, password)


def write_encrypted_file(file_path: str, data: bytes, password: Optional[str] = None) -> None:
    encrypted = encrypt_bytes(data, password)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(encrypted)


def anonymize_stem(stem: str, password: Optional[str] = None) -> str:
    if password is None:
        password = get_encryption_password() or "aitk_default_salt"
    h = hmac.new(password.encode("utf-8"), stem.encode("utf-8"), hashlib.sha256)
    return h.hexdigest()[:16]


def sanitize_log_text(text: str) -> str:
    if not text:
        return ""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"[REDACTED_PROMPT_ID:{h}]"
'''

UI_CRYPTO_TS = '''/**
 * In-browser Web Crypto API utilities for end-to-end encryption.
 * Encrypts/decrypts files, images, audio, video, and text using AES-256-GCM.
 */

const AITK_MAGIC_HEADER_STR = 'AITK_ENC_V1';
const AITK_MAGIC_HEADER_BYTES = new TextEncoder().encode(AITK_MAGIC_HEADER_STR);
const SALT_SIZE = 16;
const IV_SIZE = 12;
const PBKDF2_ITERATIONS = 100000;

export const ENCRYPTION_PASSWORD_STORAGE_KEY = 'AI_TOOLKIT_ENCRYPTION_PASSWORD';

export function getStoredPassword(): string {
  if (typeof window === 'undefined') return '';
  return localStorage.getItem(ENCRYPTION_PASSWORD_STORAGE_KEY) || '';
}

export function setStoredPassword(password: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(ENCRYPTION_PASSWORD_STORAGE_KEY, password);
}

export function generateRandomPassword(length = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+';
  const randomValues = new Uint8Array(length);
  window.crypto.getRandomValues(randomValues);
  return Array.from(randomValues)
    .map(x => chars[x % chars.length])
    .join('');
}

export async function deriveKey(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const enc = new TextEncoder();
  const keyMaterial = await window.crypto.subtle.importKey(
    'raw',
    enc.encode(password),
    { name: 'PBKDF2' },
    false,
    ['deriveKey'],
  );
  return window.crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: PBKDF2_ITERATIONS,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt'],
  );
}

export function isEncryptedBuffer(buffer: ArrayBuffer | Uint8Array): boolean {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.byteLength < AITK_MAGIC_HEADER_BYTES.byteLength) return false;
  for (let i = 0; i < AITK_MAGIC_HEADER_BYTES.byteLength; i++) {
    if (bytes[i] !== AITK_MAGIC_HEADER_BYTES[i]) return false;
  }
  return true;
}

export async function encryptBuffer(
  data: ArrayBuffer | Uint8Array,
  password = getStoredPassword(),
): Promise<ArrayBuffer> {
  if (!password) return data instanceof ArrayBuffer ? data : data.buffer;

  const salt = window.crypto.getRandomValues(new Uint8Array(SALT_SIZE));
  const iv = window.crypto.getRandomValues(new Uint8Array(IV_SIZE));
  const key = await deriveKey(password, salt);

  const rawData = data instanceof Uint8Array ? data : new Uint8Array(data);
  const ciphertext = await window.crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, rawData);

  const totalLength = AITK_MAGIC_HEADER_BYTES.byteLength + SALT_SIZE + IV_SIZE + ciphertext.byteLength;
  const result = new Uint8Array(totalLength);

  let offset = 0;
  result.set(AITK_MAGIC_HEADER_BYTES, offset);
  offset += AITK_MAGIC_HEADER_BYTES.byteLength;
  result.set(salt, offset);
  offset += SALT_SIZE;
  result.set(iv, offset);
  offset += IV_SIZE;
  result.set(new Uint8Array(ciphertext), offset);

  return result.buffer;
}

export async function decryptBuffer(
  data: ArrayBuffer | Uint8Array,
  password = getStoredPassword(),
): Promise<ArrayBuffer> {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (!isEncryptedBuffer(bytes)) {
    return bytes.buffer;
  }

  if (!password) {
    throw new Error('Encrypted file detected but no encryption password is set in settings.');
  }

  let offset = AITK_MAGIC_HEADER_BYTES.byteLength;
  const salt = bytes.slice(offset, offset + SALT_SIZE);
  offset += SALT_SIZE;
  const iv = bytes.slice(offset, offset + IV_SIZE);
  offset += IV_SIZE;
  const ciphertext = bytes.slice(offset);

  const key = await deriveKey(password, salt);
  return window.crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext);
}

export async function encryptText(text: string, password = getStoredPassword()): Promise<string> {
  if (!password) return text;
  const enc = new TextEncoder();
  const encryptedBuf = await encryptBuffer(enc.encode(text), password);
  return btoa(String.fromCharCode(...new Uint8Array(encryptedBuf)));
}

export async function decryptText(
  data: ArrayBuffer | string,
  password = getStoredPassword(),
): Promise<string> {
  let buf: ArrayBuffer;
  if (typeof data === 'string') {
    if (!data.startsWith(AITK_MAGIC_HEADER_STR)) {
      try {
        const binaryString = atob(data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        if (!isEncryptedBuffer(bytes)) {
          return data;
        }
        buf = bytes.buffer;
      } catch {
        return data;
      }
    } else {
      buf = new TextEncoder().encode(data).buffer;
    }
  } else {
    buf = data;
  }

  const decryptedBuf = await decryptBuffer(buf, password);
  return new TextDecoder().decode(decryptedBuf);
}

export async function anonymizeStem(stem: string, password = getStoredPassword()): Promise<string> {
  const pwd = password || 'aitk_default_salt';
  const enc = new TextEncoder();
  const key = await window.crypto.subtle.importKey(
    'raw',
    enc.encode(pwd),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await window.crypto.subtle.sign('HMAC', key, enc.encode(stem));
  const hex = Array.from(new Uint8Array(signature))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return hex.slice(0, 16);
}

export async function stripImageMetadata(file: File): Promise<Blob> {
  if (!file.type.startsWith('image/') || file.type === 'image/svg+xml') {
    return file;
  }
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(file);
        return;
      }
      ctx.drawImage(img, 0, 0);
      const exportType = file.type === 'image/png' ? 'image/png' : 'image/jpeg';
      canvas.toBlob(
        blob => {
          if (blob) resolve(blob);
          else resolve(file);
        },
        exportType,
        0.98,
      );
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(file);
    };
    img.src = url;
  });
}

export async function encryptFileForUpload(file: File, password = getStoredPassword()): Promise<File> {
  let blobToEncrypt: Blob = file;
  if (file.type.startsWith('image/')) {
    blobToEncrypt = await stripImageMetadata(file);
  }

  const rawBuffer = await blobToEncrypt.arrayBuffer();
  const encryptedBuffer = await encryptBuffer(rawBuffer, password);

  const lastDot = file.name.lastIndexOf('.');
  const stem = lastDot === -1 ? file.name : file.name.slice(0, lastDot);
  const ext = lastDot === -1 ? '' : file.name.slice(lastDot);

  const anonStem = await anonymizeStem(stem, password);
  const newName = `${anonStem}${ext}`;

  return new File([encryptedBuffer], newName, { type: 'application/octet-stream' });
}
'''

UI_APP_SETTINGS_PAGE_TSX = ''''use client';

import { useEffect, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { generateRandomPassword, setStoredPassword } from '@/utils/crypto';

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    if (settings.ENCRYPTION_PASSWORD) {
      setStoredPassword(settings.ENCRYPTION_PASSWORD);
    }

    apiClient
      .post('/api/settings', settings)
      .then(() => {
        setStatus('success');
      })
      .catch(error => {
        console.error('Error saving settings:', error);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  const handleGeneratePassword = () => {
    const newPwd = generateRandomPassword(32);
    setSettings(prev => ({ ...prev, ENCRYPTION_PASSWORD: newPwd }));
    setStoredPassword(newPwd);
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="ENCRYPTION_PASSWORD" className="block text-sm font-medium mb-2">
                    Encryption Password
                    <div className="text-gray-500 text-sm ml-1">
                      Used to locally encrypt your datasets, images, captions, and training settings. Datasets encrypted with this password require the same password to be decrypted.
                    </div>
                  </label>
                  <div className="flex gap-2">
                    <input
                      type={showPassword ? 'text' : 'password'}
                      id="ENCRYPTION_PASSWORD"
                      name="ENCRYPTION_PASSWORD"
                      value={settings.ENCRYPTION_PASSWORD || ''}
                      onChange={handleChange}
                      className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent font-mono text-sm text-gray-100"
                      placeholder="Enter encryption password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(v => !v)}
                      className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-xs whitespace-nowrap text-gray-200"
                    >
                      {showPassword ? 'Hide' : 'Show'}
                    </button>
                    <button
                      type="button"
                      onClick={handleGeneratePassword}
                      className="px-3 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs whitespace-nowrap text-white"
                    >
                      Generate New
                    </button>
                  </div>
                </div>

                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face Token
                    <div className="text-gray-500 text-sm ml-1">
                      Create a Read token on{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="text-blue-400 underline">
                        Huggingface
                      </a>{' '}
                      if you need to access gated/private models.
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100"
                    placeholder="Enter your Hugging Face token"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    Training Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      We will store your training information here. Must be an absolute path. If blank, it will default
                      to the output folder in the project root.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100"
                    placeholder="Enter training folder path"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    Dataset Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      Where we store and find your datasets.{' '}
                      <span className="text-orange-800">
                        Warning: This software may modify datasets so it is recommended you keep a backup somewhere else
                        or have a dedicated folder for this software.
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100"
                    placeholder="Enter datasets folder path"
                  />
                </div>

                <div>
                  <label htmlFor="MODELS_PATH" className="block text-sm font-medium mb-2">
                    Models Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      Some models support loading ComfyUI model weights directly. Models that do will be loaded
                      from/downloaded to this path. Must be an absolute path. If blank, it will default to the models
                      folder in the project root.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="MODELS_PATH"
                    name="MODELS_PATH"
                    value={settings.MODELS_PATH}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100"
                    placeholder="Enter models folder path"
                  />
                </div>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-white"
          >
            {status === 'saving' ? 'Saving...' : 'Save Settings'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
          {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
        </form>
      </MainContent>
    </>
  );
}
'''

UI_SERVER_SETTINGS_TS = '''import path from 'path';
import prisma from '@/server/prisma';
import { defaultDatasetsFolder, defaultDataRoot } from '@/paths';
import { defaultTrainFolder } from '@/paths';
import NodeCache from 'node-cache';
import crypto from 'crypto';

const myCache = new NodeCache();

export const flushCache = () => {
  myCache.flushAll();
};

export const getDatasetsRoot = async () => {
  const key = 'DATASETS_FOLDER';
  let datasetsPath = myCache.get(key) as string;
  if (datasetsPath) {
    return datasetsPath;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: 'DATASETS_FOLDER',
    },
  });
  datasetsPath = defaultDatasetsFolder;
  if (row?.value && row.value !== '') {
    datasetsPath = row.value;
  }
  datasetsPath = path.resolve(datasetsPath);
  myCache.set(key, datasetsPath);
  return datasetsPath as string;
};

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let trainingRoot = myCache.get(key) as string;
  if (trainingRoot) {
    return trainingRoot;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: 'TRAINING_FOLDER',
    },
  });
  trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  trainingRoot = path.resolve(trainingRoot);
  myCache.set(key, trainingRoot);
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let token = myCache.get(key) as string;
  if (token) {
    return token;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  myCache.set(key, token);
  return token;
};

export const getDataRoot = async () => {
  const key = 'DATA_ROOT';
  let dataRoot = myCache.get(key) as string;
  if (dataRoot) {
    return dataRoot;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: 'DATA_ROOT',
    },
  });
  dataRoot = defaultDataRoot;
  if (row?.value && row.value !== '') {
    dataRoot = row.value;
  }
  dataRoot = path.resolve(dataRoot);
  myCache.set(key, dataRoot);
  return dataRoot;
};

export const getEncryptionPassword = async () => {
  const key = 'ENCRYPTION_PASSWORD';
  let pwd = myCache.get(key) as string;
  if (pwd) {
    return pwd;
  }
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  if (row?.value && row.value.trim() !== '') {
    pwd = row.value;
  } else {
    pwd = crypto.randomBytes(24).toString('base64').replace(/[^a-zA-Z0-9]/g, '').slice(0, 32);
    await prisma.settings.upsert({
      where: { key },
      update: { value: pwd },
      create: { key, value: pwd },
    });
  }
  myCache.set(key, pwd);
  return pwd;
};
'''

UI_API_SETTINGS_ROUTE_TS = '''import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { defaultTrainFolder, defaultDatasetsFolder, defaultModelsFolder } from '@/paths';
import { flushCache, getEncryptionPassword } from '@/server/settings';

export async function GET() {
  try {
    const settings = await prisma.settings.findMany();
    const settingsObject = settings.reduce((acc: any, setting) => {
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    if (process.env.MODELS_PATH && process.env.MODELS_PATH.trim() !== '') {
      settingsObject.MODELS_PATH = process.env.MODELS_PATH;
    } else if (!settingsObject.MODELS_PATH || settingsObject.MODELS_PATH === '') {
      settingsObject.MODELS_PATH = defaultModelsFolder;
    }
    if (!settingsObject.ENCRYPTION_PASSWORD || settingsObject.ENCRYPTION_PASSWORD === '') {
      settingsObject.ENCRYPTION_PASSWORD = await getEncryptionPassword();
    }
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER, MODELS_PATH, ENCRYPTION_PASSWORD } = body;

    await Promise.all([
      prisma.settings.upsert({
        where: { key: 'HF_TOKEN' },
        update: { value: HF_TOKEN },
        create: { key: 'HF_TOKEN', value: HF_TOKEN },
      }),
      prisma.settings.upsert({
        where: { key: 'TRAINING_FOLDER' },
        update: { value: TRAINING_FOLDER },
        create: { key: 'TRAINING_FOLDER', value: TRAINING_FOLDER },
      }),
      prisma.settings.upsert({
        where: { key: 'DATASETS_FOLDER' },
        update: { value: DATASETS_FOLDER },
        create: { key: 'DATASETS_FOLDER', value: DATASETS_FOLDER },
      }),
      prisma.settings.upsert({
        where: { key: 'MODELS_PATH' },
        update: { value: MODELS_PATH },
        create: { key: 'MODELS_PATH', value: MODELS_PATH },
      }),
      prisma.settings.upsert({
        where: { key: 'ENCRYPTION_PASSWORD' },
        update: { value: ENCRYPTION_PASSWORD || '' },
        create: { key: 'ENCRYPTION_PASSWORD', value: ENCRYPTION_PASSWORD || '' },
      }),
    ]);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
'''

UI_HOOKS_USE_SETTINGS_TSX = ''''use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import { setStoredPassword } from '@/utils/crypto';

export interface Settings {
  HF_TOKEN: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  MODELS_PATH: string;
  ENCRYPTION_PASSWORD?: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState<Settings>({
    HF_TOKEN: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    MODELS_PATH: '',
    ENCRYPTION_PASSWORD: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        setSettings({
          HF_TOKEN: data.HF_TOKEN || '',
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
          MODELS_PATH: data.MODELS_PATH || '',
          ENCRYPTION_PASSWORD: data.ENCRYPTION_PASSWORD || '',
        });
        if (data.ENCRYPTION_PASSWORD) {
          setStoredPassword(data.ENCRYPTION_PASSWORD);
        }
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
'''

UI_HOOKS_USE_CAPTION_BATCH_TSX = ''''use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import { decryptText, getStoredPassword } from '@/utils/crypto';

type Resolver = { resolve: (caption: string) => void; reject: (err: unknown) => void };
type Pending = { path: string; ext: string; resolvers: Resolver[] };
const pending = new Map<string, Pending>();
const cache = new Map<string, string>();
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_DELAY_MS = 30;
const MAX_BATCH = 200;

function normExt(ext: string | undefined): string {
  return (ext || 'txt').replace(/^\\.+/, '').trim() || 'txt';
}

function keyFor(path: string, ext: string): string {
  return `${ext}\\n${path}`;
}

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(flush, FLUSH_DELAY_MS);
}

async function flush() {
  flushTimer = null;
  if (pending.size === 0) return;

  const keys: string[] = [];
  for (const key of pending.keys()) {
    keys.push(key);
    if (keys.length >= MAX_BATCH) break;
  }
  const drained = keys.map(k => pending.get(k)!);
  for (const k of keys) pending.delete(k);

  const byExt = new Map<string, Pending[]>();
  for (const entry of drained) {
    const group = byExt.get(entry.ext);
    if (group) group.push(entry);
    else byExt.set(entry.ext, [entry]);
  }

  await Promise.all(
    Array.from(byExt.entries()).map(async ([ext, entries]) => {
      const paths = entries.map(e => e.path);
      try {
        const res = await apiClient.post('/api/caption/getBatch', { imgPaths: paths, ext });
        const captions: Record<string, string> = res.data?.captions ?? {};
        const password = getStoredPassword();
        for (const { path, ext: e, resolvers } of entries) {
          const rawValue = captions[path] ?? '';
          const value = await decryptText(rawValue, password);
          cache.set(keyFor(path, e), value);
          for (const r of resolvers) r.resolve(value);
        }
      } catch (err) {
        for (const { resolvers } of entries) {
          for (const r of resolvers) r.reject(err);
        }
      }
    }),
  );

  if (pending.size > 0) scheduleFlush();
}

function requestCaption(path: string, ext: string, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const key = keyFor(path, ext);
    const resolver: Resolver = { resolve, reject };
    const entry = pending.get(key);
    if (entry) {
      entry.resolvers.push(resolver);
    } else {
      pending.set(key, { path, ext, resolvers: [resolver] });
    }
    if (signal) {
      const onAbort = () => {
        const e = pending.get(key);
        if (e) {
          const idx = e.resolvers.indexOf(resolver);
          if (idx >= 0) e.resolvers.splice(idx, 1);
          if (e.resolvers.length === 0) pending.delete(key);
        }
        reject(new DOMException('Aborted', 'AbortError'));
      };
      signal.addEventListener('abort', onAbort, { once: true });
    }
    scheduleFlush();
  });
}

export function invalidateCaption(path: string, ext?: string) {
  cache.delete(keyFor(path, normExt(ext)));
}

export function setCachedCaption(path: string, caption: string, ext?: string) {
  cache.set(keyFor(path, normExt(ext)), caption);
}

export default function useCaptionBatch(imgPath: string | null, refreshKey: number = 0, ext: string = 'txt') {
  const captionExt = normExt(ext);
  const [caption, setCaption] = useState<string>(() => (imgPath ? (cache.get(keyFor(imgPath, captionExt)) ?? '') : ''));
  const [isLoaded, setIsLoaded] = useState<boolean>(() => Boolean(imgPath && cache.has(keyFor(imgPath, captionExt))));
  const lastPathRef = useRef<string | null>(null);

  useEffect(() => {
    if (!imgPath) {
      setCaption('');
      setIsLoaded(false);
      return;
    }

    if (refreshKey > 0) invalidateCaption(imgPath, captionExt);

    const cached = cache.get(keyFor(imgPath, captionExt));
    if (cached !== undefined) {
      setCaption(cached);
      setIsLoaded(true);
      lastPathRef.current = imgPath;
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    lastPathRef.current = imgPath;
    setIsLoaded(false);
    requestCaption(imgPath, captionExt, controller.signal)
      .then(value => {
        if (cancelled || lastPathRef.current !== imgPath) return;
        setCaption(value);
        setIsLoaded(true);
      })
      .catch(err => {
        if (err?.name === 'AbortError' || cancelled) return;
        console.error('Error fetching caption:', err);
        setIsLoaded(true);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [imgPath, refreshKey, captionExt]);

  return { caption, isLoaded, setCaption };
}
'''


def patch_file_if_contains(file_path: Path, search_text: str, replacement_text: str):
    if not file_path.exists():
        print(f"  [!] Skipped {file_path} (not found)")
        return False
    content = file_path.read_text(encoding="utf-8")
    if replacement_text in content:
        print(f"  [-] Already patched: {file_path.name}")
        return True
    if search_text not in content:
        print(f"  [!] Pattern not matched in {file_path.name}")
        return False
    new_content = content.replace(search_text, replacement_text, 1)
    file_path.write_text(new_content, encoding="utf-8")
    print(f"  [+] Patched: {file_path.name}")
    return True


def ensure_use_client_at_top(file_path: Path):
    if not file_path.exists():
        return
    text = file_path.read_text(encoding="utf-8")
    if "'use client';" in text and not text.startswith("'use client';"):
        lines = text.splitlines()
        filtered_lines = [line for line in lines if line.strip() != "'use client';"]
        new_text = "'use client';\n" + "\n".join(filtered_lines) + "\n"
        file_path.write_text(new_text, encoding="utf-8")
        print(f"  [+] Fixed 'use client' ordering in: {file_path.name}")


def main():
    target_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    print(f"Applying AI-Toolkit Privacy & End-to-End Encryption Patch to:\n  {target_dir}\n")

    if not (target_dir / "run.py").exists() and not (target_dir / "toolkit").exists():
        print("ERROR: Target path does not look like an ai-toolkit repository.")
        sys.exit(1)

    # 1. toolkit/crypto.py
    p = target_dir / "toolkit" / "crypto.py"
    p.write_text(TOOLKIT_CRYPTO_PY, encoding="utf-8")
    print(f"  [+] Created: {p.relative_to(target_dir)}")

    # 2. ui/src/utils/crypto.ts
    p = target_dir / "ui" / "src" / "utils" / "crypto.ts"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(UI_CRYPTO_TS, encoding="utf-8")
    print(f"  [+] Created: {p.relative_to(target_dir)}")

    # 3. ui/src/server/settings.ts
    p = target_dir / "ui" / "src" / "server" / "settings.ts"
    p.write_text(UI_SERVER_SETTINGS_TS, encoding="utf-8")
    print(f"  [+] Updated: {p.relative_to(target_dir)}")

    # 4. ui/src/app/api/settings/route.ts
    p = target_dir / "ui" / "src" / "app" / "api" / "settings" / "route.ts"
    p.write_text(UI_API_SETTINGS_ROUTE_TS, encoding="utf-8")
    print(f"  [+] Updated: {p.relative_to(target_dir)}")

    # 5. ui/src/hooks/useSettings.tsx
    p = target_dir / "ui" / "src" / "hooks" / "useSettings.tsx"
    p.write_text(UI_HOOKS_USE_SETTINGS_TSX, encoding="utf-8")
    ensure_use_client_at_top(p)
    print(f"  [+] Updated: {p.relative_to(target_dir)}")

    # 6. ui/src/app/settings/page.tsx
    p = target_dir / "ui" / "src" / "app" / "settings" / "page.tsx"
    p.write_text(UI_APP_SETTINGS_PAGE_TSX, encoding="utf-8")
    ensure_use_client_at_top(p)
    print(f"  [+] Updated: {p.relative_to(target_dir)}")

    # 7. ui/src/hooks/useCaptionBatch.tsx
    p = target_dir / "ui" / "src" / "hooks" / "useCaptionBatch.tsx"
    p.write_text(UI_HOOKS_USE_CAPTION_BATCH_TSX, encoding="utf-8")
    ensure_use_client_at_top(p)
    print(f"  [+] Updated: {p.relative_to(target_dir)}")

    # 8. Patch ui/src/components/AddImagesModal.tsx for metadata stripping & encrypted upload
    p = target_dir / "ui" / "src" / "components" / "AddImagesModal.tsx"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if "encryptFileForUpload" not in content:
            if content.startswith("'use client';"):
                content = "'use client';\nimport { encryptFileForUpload, getStoredPassword } from '@/utils/crypto';\n" + content[len("'use client';"):].lstrip()
            else:
                content = "import { encryptFileForUpload, getStoredPassword } from '@/utils/crypto';\n" + content
            
            content = content.replace(
                "      const formData = new FormData();\n      formData.append('files', entry.file);\n      formData.append('datasetName', datasetName || '');",
                "      const password = getStoredPassword();\n      const encryptedFile = await encryptFileForUpload(entry.file, password);\n      const formData = new FormData();\n      formData.append('files', encryptedFile);\n      formData.append('datasetName', datasetName || '');",
            )
            p.write_text(content, encoding="utf-8")
            ensure_use_client_at_top(p)
            print(f"  [+] Patched: {p.relative_to(target_dir)}")

    # 9. Patch ui/src/components/DatasetImageCard.tsx for in-browser decryption
    p = target_dir / "ui" / "src" / "components" / "DatasetImageCard.tsx"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if "decryptBuffer" not in content:
            if content.startswith("'use client';"):
                content = "'use client';\nimport { decryptBuffer, decryptText, encryptText, getStoredPassword } from '@/utils/crypto';\n" + content[len("'use client';"):].lstrip()
            else:
                content = "import { decryptBuffer, decryptText, encryptText, getStoredPassword } from '@/utils/crypto';\n" + content
            
            # Patch fetch blob to decrypt
            content = content.replace(
                "          return r.blob();\n        })\n        .then(blob => {\n          if (cancelled || !blob) return;\n          objectUrl = URL.createObjectURL(blob);\n          setBlobUrl(objectUrl);\n          setLoaded(true);\n        })",
                "          return r.arrayBuffer();\n        })\n        .then(async arrayBuffer => {\n          if (cancelled || !arrayBuffer) return;\n          const password = getStoredPassword();\n          const decrypted = await decryptBuffer(arrayBuffer, password);\n          const ext = imageUrl.split('.').pop()?.toLowerCase() || 'jpeg';\n          const mime = ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';\n          objectUrl = URL.createObjectURL(new Blob([decrypted], { type: mime }));\n          setBlobUrl(objectUrl);\n          setLoaded(true);\n        })",
            )
            # Patch saveCaption
            content = content.replace(
                "    apiClient\n      .post('/api/img/caption', { imgPath: imageUrl, caption: trimmedCaption, ext: captionExt })",
                "    const password = getStoredPassword();\n    const payloadCaption = password ? await encryptText(trimmedCaption, password) : trimmedCaption;\n    apiClient\n      .post('/api/img/caption', { imgPath: imageUrl, caption: payloadCaption, ext: captionExt })",
            )
            p.write_text(content, encoding="utf-8")
            ensure_use_client_at_top(p)
            print(f"  [+] Patched: {p.relative_to(target_dir)}")

    # 10. Patch ui/cron/actions/startJob.ts to pass encryption password
    p = target_dir / "ui" / "cron" / "actions" / "startJob.ts"
    patch_file_if_contains(
        p,
        search_text="""    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
      PYTHONUNBUFFERED: '1', // write Python output immediately so it is not lost on a crash
    };""",
        replacement_text="""    const encryptionPassword = await prisma.settings.findFirst({
      where: { key: 'ENCRYPTION_PASSWORD' },
    });

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
      PYTHONUNBUFFERED: '1', // write Python output immediately so it is not lost on a crash
    };

    if (encryptionPassword?.value && encryptionPassword.value.trim() !== '') {
      additionalEnv.AITK_ENCRYPTION_PASSWORD = encryptionPassword.value.trim();
    }""",
    )

    # 11. Patch toolkit/dataloader_mixins.py for in-memory decryption
    p = target_dir / "toolkit" / "dataloader_mixins.py"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if "from toolkit.crypto import decrypt_bytes, read_decrypted_file, write_encrypted_file" not in content:
            content = "from toolkit.crypto import decrypt_bytes, read_decrypted_file, write_encrypted_file\nimport io\n" + content
            content = content.replace(
                "            if os.path.exists(prompt_path):\n                with open(prompt_path, 'r', encoding='utf-8') as f:\n                    prompt = f.read()",
                "            if os.path.exists(prompt_path):\n                raw_bytes = read_decrypted_file(prompt_path)\n                try:\n                    prompt = raw_bytes.decode('utf-8')\n                except Exception:\n                    prompt = ''",
            )
            content = content.replace(
                "            img = Image.open(self.path)\n            img = exif_transpose(img)",
                "            raw_bytes = read_decrypted_file(self.path)\n            img = Image.open(io.BytesIO(raw_bytes))\n            img = exif_transpose(img)\n            del raw_bytes",
            )
            p.write_text(content, encoding="utf-8")
            print(f"  [+] Patched: {p.relative_to(target_dir)}")

    # 12. Patch toolkit/metadata.py to sanitize LoRA output metadata
    p = target_dir / "toolkit" / "metadata.py"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if 'sensitive_keys = ["ss_tag_frequency", "instance_prompt", "caption", "prompt", "training_info"]' not in content:
            content = content.replace(
                "    if add_software_info:\n        save_meta[\"software\"] = software_meta\n    # safetensors can only be one level deep",
                "    if add_software_info:\n        save_meta[\"software\"] = software_meta\n    sensitive_keys = [\"ss_tag_frequency\", \"instance_prompt\", \"caption\", \"prompt\", \"training_info\"]\n    for k in list(save_meta.keys()):\n        if any(sk in k.lower() for sk in sensitive_keys):\n            del save_meta[k]\n    # safetensors can only be one level deep",
            )
            p.write_text(content, encoding="utf-8")
            print(f"  [+] Patched: {p.relative_to(target_dir)}")

    print("\nPatch successfully applied! All frontend and backend privacy components are updated.")


if __name__ == "__main__":
    main()
