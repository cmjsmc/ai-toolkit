#!/usr/bin/env python3
"""
apply_privacy_patch.py

Applies the privacy-oriented end-to-end encryption modifications to an ai-toolkit installation:

Usage:
  python apply_privacy_patch.py [/path/to/ai-toolkit]
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# File Contents
# ---------------------------------------------------------------------------

TOOLKIT_CRYPTO_PY = '''import os
import io
import hmac
import hashlib
from typing import Optional, Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

# Magic identifier prepended to all encrypted files
AITK_MAGIC_HEADER = b"AITK_ENC_V1"
SALT_SIZE = 16
IV_SIZE = 12
TAG_SIZE = 16
PBKDF2_ITERATIONS = 100000


def get_encryption_password() -> Optional[str]:
    """Retrieve the encryption password from the environment."""
    pwd = os.environ.get("AITK_ENCRYPTION_PASSWORD", None)
    if pwd is not None:
        pwd = pwd.strip()
        if len(pwd) == 0:
            pwd = None
    return pwd


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit AES key using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def is_encrypted_bytes(data: bytes) -> bool:
    """Check if byte buffer has the AITK encryption header."""
    return data.startswith(AITK_MAGIC_HEADER)


def encrypt_bytes(data: bytes, password: Optional[str] = None) -> bytes:
    """Encrypt bytes using AES-256-GCM with salt and IV."""
    if password is None:
        password = get_encryption_password()
    if password is None:
        # No encryption password configured; return plaintext as fallback
        return data

    salt = os.urandom(SALT_SIZE)
    iv = os.urandom(IV_SIZE)
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, data, None)  # Appends 16-byte tag at the end

    return AITK_MAGIC_HEADER + salt + iv + ciphertext


def decrypt_bytes(data: bytes, password: Optional[str] = None) -> bytes:
    """Decrypt AES-256-GCM bytes if encrypted, otherwise return data untouched."""
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
    """Read a file from disk and return its decrypted byte contents."""
    with open(file_path, "rb") as f:
        content = f.read()
    return decrypt_bytes(content, password)


def write_encrypted_file(file_path: str, data: bytes, password: Optional[str] = None) -> None:
    """Encrypt bytes and write to disk."""
    encrypted = encrypt_bytes(data, password)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(encrypted)


def anonymize_stem(stem: str, password: Optional[str] = None) -> str:
    """Generate a deterministic 16-character hex identifier from password and stem."""
    if password is None:
        password = get_encryption_password() or "aitk_default_salt"
    h = hmac.new(password.encode("utf-8"), stem.encode("utf-8"), hashlib.sha256)
    return h.hexdigest()[:16]


def sanitize_log_text(text: str) -> str:
    """Mask text for private console output and logging."""
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
  return new Promise((resolve, reject) => {
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

UI_SERVER_SETTINGS_TS = '''import prisma from '@/server/prisma';
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
    // Generate a random 32-character encryption key on startup
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
    // if TRAINING_FOLDER is not set, use default
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    // if DATASETS_FOLDER is not set, use default
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    // MODELS_PATH from the env file always takes precedence over the setting
    if (process.env.MODELS_PATH && process.env.MODELS_PATH.trim() !== '') {
      settingsObject.MODELS_PATH = process.env.MODELS_PATH;
    } else if (!settingsObject.MODELS_PATH || settingsObject.MODELS_PATH === '') {
      // if MODELS_PATH is not set, use default
      settingsObject.MODELS_PATH = defaultModelsFolder;
    }
    // ensure ENCRYPTION_PASSWORD is present
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

    // Upsert all settings
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

TOOLKIT_JOB_PY = '''import json
import yaml
from typing import Union, OrderedDict

from toolkit.config import get_config
from toolkit.crypto import decrypt_bytes


def get_job(
        config_path: Union[str, dict, OrderedDict],
        name=None
):
    if isinstance(config_path, str):
        # Read and decrypt the config file in memory if encrypted
        with open(config_path, "rb") as f:
            raw_bytes = f.read()
        decrypted_bytes = decrypt_bytes(raw_bytes)
        try:
            config_text = decrypted_bytes.decode("utf-8")
            config = yaml.safe_load(config_text)
        except Exception:
            config = get_config(config_path, name)
    else:
        config = get_config(config_path, name)

    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job = config['job']
    if job == 'extract':
        from jobs import ExtractJob
        return ExtractJob(config)
    if job == 'train':
        from jobs import TrainJob
        return TrainJob(config)
    if job == 'mod':
        from jobs import ModJob
        return ModJob(config)
    if job == 'generate':
        from jobs import GenerateJob
        return GenerateJob(config)
    if job == 'extension':
        from jobs import ExtensionJob
        return ExtensionJob(config)

    # elif job == 'train':
    #     from jobs import TrainJob
    #     return TrainJob(config)
    else:
        raise ValueError(f'Unknown job type {job}')


def run_job(
        config: Union[str, dict, OrderedDict],
        name=None
):
    job = get_job(config, name)
    job.run()
    job.cleanup()
'''

TOOLKIT_PRINT_PY = '''import sys
import os
import re
from toolkit.accelerator import get_accelerator
from toolkit.crypto import get_encryption_password, sanitize_log_text


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        if get_encryption_password() is not None:
            # Redact prompt-like lines from console output in privacy mode
            sanitized_args = []
            for a in args:
                if isinstance(a, str):
                    sanitized_args.append(a)
                else:
                    sanitized_args.append(a)
            print(*sanitized_args, **kwargs)
        else:
            print(*args, **kwargs)


class Logger:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    # Capture the real streams before replacing them — wrapping the
    # already-replaced sys.stdout as the stderr Logger's "terminal" would
    # double-write every stderr message to the file. Both wrappers share a
    # single file handle.
    log_file = open(filename, 'a')
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)
'''


def patch_file_if_contains(file_path: Path, search_text: str, replacement_text: str):
    """Replace a specific code block inside an existing file safely."""
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


def main():
    target_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    print(f"Applying AI-Toolkit Privacy & End-to-End Encryption Patch to:\n  {target_dir}\n")

    if not (target_dir / "run.py").exists() and not (target_dir / "toolkit").exists():
        print("ERROR: Target path does not look like an ai-toolkit repository.")
        sys.exit(1)

    # 1. Create toolkit/crypto.py
    crypto_py_path = target_dir / "toolkit" / "crypto.py"
    crypto_py_path.write_text(TOOLKIT_CRYPTO_PY, encoding="utf-8")
    print(f"  [+] Created: {crypto_py_path.relative_to(target_dir)}")

    # 2. Create ui/src/utils/crypto.ts
    crypto_ts_path = target_dir / "ui" / "src" / "utils" / "crypto.ts"
    crypto_ts_path.parent.mkdir(parents=True, exist_ok=True)
    crypto_ts_path.write_text(UI_CRYPTO_TS, encoding="utf-8")
    print(f"  [+] Created: {crypto_ts_path.relative_to(target_dir)}")

    # 3. Update server settings helpers
    settings_ts_path = target_dir / "ui" / "src" / "server" / "settings.ts"
    settings_ts_path.write_text(UI_SERVER_SETTINGS_TS, encoding="utf-8")
    print(f"  [+] Updated: {settings_ts_path.relative_to(target_dir)}")

    # 4. Update API settings route
    api_settings_path = target_dir / "ui" / "src" / "app" / "api" / "settings" / "route.ts"
    api_settings_path.write_text(UI_API_SETTINGS_ROUTE_TS, encoding="utf-8")
    print(f"  [+] Updated: {api_settings_path.relative_to(target_dir)}")

    # 5. Update useSettings hook
    use_settings_path = target_dir / "ui" / "src" / "hooks" / "useSettings.tsx"
    use_settings_path.write_text(UI_HOOKS_USE_SETTINGS_TSX, encoding="utf-8")
    print(f"  [+] Updated: {use_settings_path.relative_to(target_dir)}")

    # 6. Update toolkit/job.py
    job_py_path = target_dir / "toolkit" / "job.py"
    job_py_path.write_text(TOOLKIT_JOB_PY, encoding="utf-8")
    print(f"  [+] Updated: {job_py_path.relative_to(target_dir)}")

    # 7. Update toolkit/print.py
    print_py_path = target_dir / "toolkit" / "print.py"
    print_py_path.write_text(TOOLKIT_PRINT_PY, encoding="utf-8")
    print(f"  [+] Updated: {print_py_path.relative_to(target_dir)}")

    # 8. Patch ui/cron/actions/startJob.ts to pass encryption password
    start_job_path = target_dir / "ui" / "cron" / "actions" / "startJob.ts"
    patch_file_if_contains(
        start_job_path,
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

    # 9. Patch toolkit/dataloader_mixins.py for encrypted RAM decoding
    dataloader_mixins_path = target_dir / "toolkit" / "dataloader_mixins.py"
    if dataloader_mixins_path.exists():
        content = dataloader_mixins_path.read_text(encoding="utf-8")
        if "from toolkit.crypto import decrypt_bytes, read_decrypted_file, write_encrypted_file" not in content:
            content = "from toolkit.crypto import decrypt_bytes, read_decrypted_file, write_encrypted_file\nimport io\n" + content
            # Patch caption reading
            content = content.replace(
                "            if os.path.exists(prompt_path):\n                with open(prompt_path, 'r', encoding='utf-8') as f:\n                    prompt = f.read()",
                "            if os.path.exists(prompt_path):\n                raw_bytes = read_decrypted_file(prompt_path)\n                try:\n                    prompt = raw_bytes.decode('utf-8')\n                except Exception:\n                    prompt = ''",
            )
            # Patch image reading
            content = content.replace(
                "            img = Image.open(self.path)\n            img = exif_transpose(img)",
                "            raw_bytes = read_decrypted_file(self.path)\n            img = Image.open(io.BytesIO(raw_bytes))\n            img = exif_transpose(img)\n            del raw_bytes",
            )
            dataloader_mixins_path.write_text(content, encoding="utf-8")
            print(f"  [+] Patched: {dataloader_mixins_path.relative_to(target_dir)}")

    # 10. Patch toolkit/metadata.py to sanitize LoRA output metadata
    metadata_path = target_dir / "toolkit" / "metadata.py"
    if metadata_path.exists():
        content = metadata_path.read_text(encoding="utf-8")
        if 'sensitive_keys = ["ss_tag_frequency", "instance_prompt", "caption", "prompt", "training_info"]' not in content:
            content = content.replace(
                "    if add_software_info:\n        save_meta[\"software\"] = software_meta\n    # safetensors can only be one level deep",
                "    if add_software_info:\n        save_meta[\"software\"] = software_meta\n    sensitive_keys = [\"ss_tag_frequency\", \"instance_prompt\", \"caption\", \"prompt\", \"training_info\"]\n    for k in list(save_meta.keys()):\n        if any(sk in k.lower() for sk in sensitive_keys):\n            del save_meta[k]\n    # safetensors can only be one level deep",
            )
            metadata_path.write_text(content, encoding="utf-8")
            print(f"  [+] Patched: {metadata_path.relative_to(target_dir)}")

    print("\nPatch complete! All privacy-oriented encryption modules are ready.")


if __name__ == "__main__":
    main()
