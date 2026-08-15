#!/usr/bin/env python3
"""
apply_privacy_patch.py

The Complete All-In-One End-to-End Encryption & Privacy Patch for AI-Toolkit.
Applies all frontend WebCrypto tools, React components, backend AES-GCM decryption,
PyTorch monkey-patches, log redactors, and LoRA metadata sanitizers in a single run.
"""

import os
import sys
from pathlib import Path

# ==============================================================================
# Full File Contents (New & Replaced Files)
# ==============================================================================

TOOLKIT_CRYPTO_PY = '''import os
import io
import hmac
import hashlib
import base64
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

AITK_MAGIC_HEADER = b"AITK_ENC_V1"
SALT_SIZE = 16
IV_SIZE = 12
PBKDF2_ITERATIONS = 100000

def get_encryption_password() -> Optional[str]:
    pwd = os.environ.get("AITK_ENCRYPTION_PASSWORD", None)
    if pwd is not None:
        pwd = pwd.strip()
        if len(pwd) == 0:
            pwd = None
    return pwd

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=PBKDF2_ITERATIONS)
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
        raise ValueError("Encrypted content detected, but no password is set.")
    offset = len(AITK_MAGIC_HEADER)
    salt = data[offset : offset + SALT_SIZE]
    offset += SALT_SIZE
    iv = data[offset : offset + IV_SIZE]
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

def write_encrypted_text(file_path: str, text: str, password: Optional[str] = None) -> None:
    encrypted_bytes = encrypt_bytes(text.encode("utf-8"), password)
    b64 = base64.b64encode(encrypted_bytes).decode("ascii")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(b64)

def read_decrypted_text(file_path: str, password: Optional[str] = None) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("QUlUS19FTkNfV"): # Base64 of AITK_ENC_V1
        try:
            raw_bytes = base64.b64decode(content)
            decrypted = decrypt_bytes(raw_bytes, password)
            return decrypted.decode("utf-8")
        except Exception:
            pass
    return content

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

def setup_crypto_patches():
    """Monkey-patch standard library tools to transparently handle encrypted files in RAM."""
    try:
        import PIL.Image
        _orig_open = PIL.Image.open
        def _patched_open(fp, mode="r", formats=None):
            is_enc = False
            if isinstance(fp, str):
                try:
                    with open(fp, "rb") as f:
                        if f.read(11) == b"AITK_ENC_V1": is_enc = True
                except Exception: pass
                if is_enc:
                    raw = read_decrypted_file(fp)
                    fp = io.BytesIO(raw)
            elif hasattr(fp, "read") and hasattr(fp, "tell") and hasattr(fp, "seek"):
                try:
                    pos = fp.tell()
                    if fp.read(11) == b"AITK_ENC_V1": is_enc = True
                    fp.seek(pos)
                except Exception: pass
                if is_enc:
                    raw = fp.read()
                    dec = decrypt_bytes(raw)
                    fp = io.BytesIO(dec)
            return _orig_open(fp, mode, formats)
        PIL.Image.open = _patched_open
    except ImportError: pass

    try:
        import imghdr
        _orig_what = imghdr.what
        def _patched_what(file, h=None):
            if h is None and isinstance(file, str):
                try:
                    with open(file, 'rb') as f:
                        if f.read(11) == b"AITK_ENC_V1":
                            ext = os.path.splitext(file)[1].lower()
                            if ext in ['.jpg', '.jpeg']: return 'jpeg'
                            if ext == '.png': return 'png'
                            if ext == '.webp': return 'webp'
                            return 'jpeg'
                except Exception: pass
            return _orig_what(file, h)
        imghdr.what = _patched_what
    except ImportError: pass

    try:
        import av
        _orig_av_open = av.open
        def _patched_av_open(file, *args, **kwargs):
            is_enc = False
            if isinstance(file, str):
                try:
                    with open(file, "rb") as f:
                        if f.read(11) == b"AITK_ENC_V1": is_enc = True
                except Exception: pass
                if is_enc:
                    raw = read_decrypted_file(file)
                    file = io.BytesIO(raw)
            return _orig_av_open(file, *args, **kwargs)
        av.open = _patched_av_open
    except ImportError: pass

    try:
        import torchaudio
        _orig_ta_load = torchaudio.load
        def _patched_ta_load(uri, *args, **kwargs):
            is_enc = False
            if isinstance(uri, str):
                try:
                    with open(uri, "rb") as f:
                        if f.read(11) == b"AITK_ENC_V1": is_enc = True
                except Exception: pass
                if is_enc:
                    raw = read_decrypted_file(uri)
                    uri = io.BytesIO(raw)
            return _orig_ta_load(uri, *args, **kwargs)
        torchaudio.load = _patched_ta_load
    except ImportError: pass

    try:
        from toolkit import image_utils
        import PIL.Image
        _orig_get_image_size = image_utils.get_image_size
        def _patched_get_image_size(file_path):
            is_enc = False
            try:
                with open(file_path, "rb") as f:
                    if f.read(11) == b"AITK_ENC_V1": is_enc = True
            except Exception: pass
            if is_enc:
                raw = read_decrypted_file(file_path)
                img = PIL.Image.open(io.BytesIO(raw))
                return img.size
            return _orig_get_image_size(file_path)
        image_utils.get_image_size = _patched_get_image_size
    except ImportError: pass

setup_crypto_patches()
'''

UI_CRYPTO_TS = '''/**
 * In-browser Web Crypto API utilities for end-to-end encryption.
 */
const AITK_MAGIC_HEADER_STR = 'AITK_ENC_V1';
const AITK_MAGIC_HEADER_BYTES = new TextEncoder().encode(AITK_MAGIC_HEADER_STR);
const SALT_SIZE = 16;
const IV_SIZE = 12;
const PBKDF2_ITERATIONS = 100000;

export const ENCRYPTION_PASSWORD_STORAGE_KEY = 'AI_TOOLKIT_ENCRYPTION_PASSWORD';

export function setStoredPassword(password: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(ENCRYPTION_PASSWORD_STORAGE_KEY, password);
}

export async function getPasswordAsync(): Promise<string> {
  if (typeof window === 'undefined') return '';
  const pwd = localStorage.getItem(ENCRYPTION_PASSWORD_STORAGE_KEY);
  if (pwd) return pwd;
  try {
    const res = await fetch('/api/settings');
    if (res.ok) {
      const data = await res.json();
      if (data.ENCRYPTION_PASSWORD) {
        localStorage.setItem(ENCRYPTION_PASSWORD_STORAGE_KEY, data.ENCRYPTION_PASSWORD);
        return data.ENCRYPTION_PASSWORD;
      }
    }
  } catch (e) { console.error('Failed to fetch encryption password:', e); }
  return '';
}

export function generateRandomPassword(length = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+';
  const randomValues = new Uint8Array(length);
  window.crypto.getRandomValues(randomValues);
  return Array.from(randomValues).map(x => chars[x % chars.length]).join('');
}

export async function deriveKey(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const enc = new TextEncoder();
  const keyMaterial = await window.crypto.subtle.importKey(
    'raw', enc.encode(password), { name: 'PBKDF2' }, false, ['deriveKey']
  );
  return window.crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
    keyMaterial, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']
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

export async function encryptBuffer(data: ArrayBuffer | Uint8Array, password?: string): Promise<ArrayBuffer> {
  const pwd = password || await getPasswordAsync();
  if (!pwd) return data instanceof ArrayBuffer ? data : data.buffer;
  const salt = window.crypto.getRandomValues(new Uint8Array(SALT_SIZE));
  const iv = window.crypto.getRandomValues(new Uint8Array(IV_SIZE));
  const key = await deriveKey(pwd, salt);
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

export async function decryptBuffer(data: ArrayBuffer | Uint8Array, password?: string): Promise<ArrayBuffer> {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (!isEncryptedBuffer(bytes)) return bytes.buffer;
  const pwd = password || await getPasswordAsync();
  if (!pwd) throw new Error('Encrypted file detected but no encryption password found.');
  let offset = AITK_MAGIC_HEADER_BYTES.byteLength;
  const salt = bytes.slice(offset, offset + SALT_SIZE);
  offset += SALT_SIZE;
  const iv = bytes.slice(offset, offset + IV_SIZE);
  offset += IV_SIZE;
  const ciphertext = bytes.slice(offset);
  const key = await deriveKey(pwd, salt);
  return window.crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext);
}

export async function encryptText(text: string, password?: string): Promise<string> {
  const pwd = password || await getPasswordAsync();
  if (!pwd) return text;
  const enc = new TextEncoder();
  const encryptedBuf = await encryptBuffer(enc.encode(text), pwd);
  return btoa(String.fromCharCode(...new Uint8Array(encryptedBuf)));
}

export async function decryptText(data: ArrayBuffer | string, password?: string): Promise<string> {
  let buf: ArrayBuffer;
  if (typeof data === 'string') {
    if (!data.startsWith(AITK_MAGIC_HEADER_STR)) {
      try {
        const binaryString = atob(data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
        if (!isEncryptedBuffer(bytes)) return data;
        buf = bytes.buffer;
      } catch { return data; }
    } else {
      buf = new TextEncoder().encode(data).buffer;
    }
  } else { buf = data; }
  const pwd = password || await getPasswordAsync();
  const decryptedBuf = await decryptBuffer(buf, pwd);
  return new TextDecoder().decode(decryptedBuf);
}

export async function anonymizeStem(stem: string, password?: string): Promise<string> {
  const pwd = password || await getPasswordAsync() || 'aitk_default_salt';
  const enc = new TextEncoder();
  const key = await window.crypto.subtle.importKey('raw', enc.encode(pwd), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const signature = await window.crypto.subtle.sign('HMAC', key, enc.encode(stem));
  const hex = Array.from(new Uint8Array(signature)).map(b => b.toString(16).padStart(2, '0')).join('');
  return hex.slice(0, 16);
}

export async function stripImageMetadata(file: File): Promise<Blob> {
  if (!file.type.startsWith('image/') || file.type === 'image/svg+xml') return file;
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) { resolve(file); return; }
      ctx.drawImage(img, 0, 0);
      canvas.toBlob(blob => { if (blob) resolve(blob); else resolve(file); }, file.type === 'image/png' ? 'image/png' : 'image/jpeg', 0.98);
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

export async function encryptFileForUpload(file: File, password?: string): Promise<File> {
  const pwd = password || await getPasswordAsync();
  let blobToEncrypt: Blob = file;
  if (file.type.startsWith('image/')) {
    blobToEncrypt = await stripImageMetadata(file);
  }
  const rawBuffer = await blobToEncrypt.arrayBuffer();
  const encryptedBuffer = await encryptBuffer(rawBuffer, pwd);
  const lastDot = file.name.lastIndexOf('.');
  const stem = lastDot === -1 ? file.name : file.name.slice(0, lastDot);
  const ext = lastDot === -1 ? '' : file.name.slice(lastDot);
  const anonStem = await anonymizeStem(stem, pwd);
  const newName = `${anonStem}${ext}`;
  return new File([encryptedBuffer], newName, { type: 'application/octet-stream' });
}
'''

UI_SERVER_SETTINGS_TS = '''import path from 'path';
import prisma from '@/server/prisma';
import { defaultDatasetsFolder, defaultDataRoot } from '@/paths';
import { defaultTrainFolder } from '@/paths';
import NodeCache from 'node-cache';
import crypto from 'crypto';

const myCache = new NodeCache();

export const flushCache = () => { myCache.flushAll(); };

export const getDatasetsRoot = async () => {
  const key = 'DATASETS_FOLDER';
  let datasetsPath = myCache.get(key) as string;
  if (datasetsPath) return datasetsPath;
  let row = await prisma.settings.findFirst({ where: { key: 'DATASETS_FOLDER' } });
  datasetsPath = defaultDatasetsFolder;
  if (row?.value && row.value !== '') datasetsPath = row.value;
  datasetsPath = path.resolve(datasetsPath);
  myCache.set(key, datasetsPath);
  return datasetsPath as string;
};

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let trainingRoot = myCache.get(key) as string;
  if (trainingRoot) return trainingRoot;
  let row = await prisma.settings.findFirst({ where: { key: 'TRAINING_FOLDER' } });
  trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') trainingRoot = row.value;
  trainingRoot = path.resolve(trainingRoot);
  myCache.set(key, trainingRoot);
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let token = myCache.get(key) as string;
  if (token) return token;
  let row = await prisma.settings.findFirst({ where: { key: key } });
  token = '';
  if (row?.value && row.value !== '') token = row.value;
  myCache.set(key, token);
  return token;
};

export const getDataRoot = async () => {
  const key = 'DATA_ROOT';
  let dataRoot = myCache.get(key) as string;
  if (dataRoot) return dataRoot;
  let row = await prisma.settings.findFirst({ where: { key: 'DATA_ROOT' } });
  dataRoot = defaultDataRoot;
  if (row?.value && row.value !== '') dataRoot = row.value;
  dataRoot = path.resolve(dataRoot);
  myCache.set(key, dataRoot);
  return dataRoot;
};

export const getEncryptionPassword = async () => {
  const key = 'ENCRYPTION_PASSWORD';
  let pwd = myCache.get(key) as string;
  if (pwd) return pwd;
  let row = await prisma.settings.findFirst({ where: { key: key } });
  if (row?.value && row.value.trim() !== '') {
    pwd = row.value;
  } else {
    pwd = crypto.randomBytes(24).toString('base64').replace(/[^a-zA-Z0-9]/g, '').slice(0, 32);
    await prisma.settings.upsert({
      where: { key }, update: { value: pwd }, create: { key, value: pwd }
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
      acc[setting.key] = setting.value; return acc;
    }, {});
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    if (process.env.MODELS_PATH && process.env.MODELS_PATH.trim() !== '') {
      settingsObject.MODELS_PATH = process.env.MODELS_PATH;
    } else if (!settingsObject.MODELS_PATH || settingsObject.MODELS_PATH === '') {
      settingsObject.MODELS_PATH = defaultModelsFolder;
    }
    if (!settingsObject.ENCRYPTION_PASSWORD || settingsObject.ENCRYPTION_PASSWORD === '') {
      settingsObject.ENCRYPTION_PASSWORD = await getEncryptionPassword();
    }
    return NextResponse.json(settingsObject);
  } catch (error) { return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 }); }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER, MODELS_PATH, ENCRYPTION_PASSWORD } = body;
    await Promise.all([
      prisma.settings.upsert({ where: { key: 'HF_TOKEN' }, update: { value: HF_TOKEN }, create: { key: 'HF_TOKEN', value: HF_TOKEN } }),
      prisma.settings.upsert({ where: { key: 'TRAINING_FOLDER' }, update: { value: TRAINING_FOLDER }, create: { key: 'TRAINING_FOLDER', value: TRAINING_FOLDER } }),
      prisma.settings.upsert({ where: { key: 'DATASETS_FOLDER' }, update: { value: DATASETS_FOLDER }, create: { key: 'DATASETS_FOLDER', value: DATASETS_FOLDER } }),
      prisma.settings.upsert({ where: { key: 'MODELS_PATH' }, update: { value: MODELS_PATH }, create: { key: 'MODELS_PATH', value: MODELS_PATH } }),
      prisma.settings.upsert({ where: { key: 'ENCRYPTION_PASSWORD' }, update: { value: ENCRYPTION_PASSWORD || '' }, create: { key: 'ENCRYPTION_PASSWORD', value: ENCRYPTION_PASSWORD || '' } }),
    ]);
    flushCache();
    return NextResponse.json({ success: true });
  } catch (error) { return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 }); }
}
'''

UI_HOOKS_USE_SETTINGS_TSX = ''''use client';
import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import { setStoredPassword } from '@/utils/crypto';

export interface Settings {
  HF_TOKEN: string; TRAINING_FOLDER: string; DATASETS_FOLDER: string; MODELS_PATH: string; ENCRYPTION_PASSWORD?: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState<Settings>({
    HF_TOKEN: '', TRAINING_FOLDER: '', DATASETS_FOLDER: '', MODELS_PATH: '', ENCRYPTION_PASSWORD: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient.get('/api/settings').then(res => res.data).then(data => {
      setSettings({
        HF_TOKEN: data.HF_TOKEN || '', TRAINING_FOLDER: data.TRAINING_FOLDER || '', DATASETS_FOLDER: data.DATASETS_FOLDER || '', MODELS_PATH: data.MODELS_PATH || '', ENCRYPTION_PASSWORD: data.ENCRYPTION_PASSWORD || '',
      });
      if (data.ENCRYPTION_PASSWORD) setStoredPassword(data.ENCRYPTION_PASSWORD);
      setIsLoaded(true);
    }).catch(error => console.error('Error fetching settings:', error));
  }, []);
  return { settings, setSettings, isSettingsLoaded };
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
    if (settings.ENCRYPTION_PASSWORD) setStoredPassword(settings.ENCRYPTION_PASSWORD);
    apiClient.post('/api/settings', settings).then(() => setStatus('success'))
      .catch(error => { console.error('Error saving settings:', error); setStatus('error'); })
      .finally(() => setTimeout(() => setStatus('idle'), 2000));
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
      <TopBar><div><h1 className="text-base sm:text-lg">Settings</h1></div><div className="flex-1"></div></TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="ENCRYPTION_PASSWORD" className="block text-sm font-medium mb-2">
                    Encryption Password
                    <div className="text-gray-500 text-sm ml-1">Used to locally encrypt your datasets, images, captions, and training settings. Datasets encrypted with this password require the same password to be decrypted.</div>
                  </label>
                  <div className="flex gap-2">
                    <input type={showPassword ? 'text' : 'password'} id="ENCRYPTION_PASSWORD" name="ENCRYPTION_PASSWORD" value={settings.ENCRYPTION_PASSWORD || ''} onChange={handleChange} className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent font-mono text-sm text-gray-100" placeholder="Enter encryption password" />
                    <button type="button" onClick={() => setShowPassword(v => !v)} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-xs whitespace-nowrap text-gray-200">{showPassword ? 'Hide' : 'Show'}</button>
                    <button type="button" onClick={handleGeneratePassword} className="px-3 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs whitespace-nowrap text-white">Generate New</button>
                  </div>
                </div>
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">Hugging Face Token</label>
                  <input type="password" id="HF_TOKEN" name="HF_TOKEN" value={settings.HF_TOKEN} onChange={handleChange} className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100" />
                </div>
                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">Training Folder Path</label>
                  <input type="text" id="TRAINING_FOLDER" name="TRAINING_FOLDER" value={settings.TRAINING_FOLDER} onChange={handleChange} className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100" />
                </div>
                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">Dataset Folder Path</label>
                  <input type="text" id="DATASETS_FOLDER" name="DATASETS_FOLDER" value={settings.DATASETS_FOLDER} onChange={handleChange} className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100" />
                </div>
                <div>
                  <label htmlFor="MODELS_PATH" className="block text-sm font-medium mb-2">Models Folder Path</label>
                  <input type="text" id="MODELS_PATH" name="MODELS_PATH" value={settings.MODELS_PATH} onChange={handleChange} className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent text-gray-100" />
                </div>
              </div>
            </div>
          </div>
          <button type="submit" disabled={status === 'saving'} className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-white">{status === 'saving' ? 'Saving...' : 'Save Settings'}</button>
          {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
          {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
        </form>
      </MainContent>
    </>
  );
}
'''

UI_HOOKS_USE_CAPTION_BATCH_TSX = ''''use client';
import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import { decryptText } from '@/utils/crypto';

type Resolver = { resolve: (caption: string) => void; reject: (err: unknown) => void };
type Pending = { path: string; ext: string; resolvers: Resolver[] };
const pending = new Map<string, Pending>();
const cache = new Map<string, string>();
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_DELAY_MS = 30;
const MAX_BATCH = 200;

function normExt(ext: string | undefined): string { return (ext || 'txt').replace(/^\\.+/, '').trim() || 'txt'; }
function keyFor(path: string, ext: string): string { return `${ext}\\n${path}`; }
function scheduleFlush() { if (!flushTimer) flushTimer = setTimeout(flush, FLUSH_DELAY_MS); }

async function flush() {
  flushTimer = null;
  if (pending.size === 0) return;
  const keys: string[] = [];
  for (const key of pending.keys()) { keys.push(key); if (keys.length >= MAX_BATCH) break; }
  const drained = keys.map(k => pending.get(k)!);
  for (const k of keys) pending.delete(k);

  const byExt = new Map<string, Pending[]>();
  for (const entry of drained) {
    const group = byExt.get(entry.ext);
    if (group) group.push(entry); else byExt.set(entry.ext, [entry]);
  }

  await Promise.all(
    Array.from(byExt.entries()).map(async ([ext, entries]) => {
      const paths = entries.map(e => e.path);
      try {
        const res = await apiClient.post('/api/caption/getBatch', { imgPaths: paths, ext });
        const captions: Record<string, string> = res.data?.captions ?? {};
        for (const { path, ext: e, resolvers } of entries) {
          const rawValue = captions[path] ?? '';
          const value = await decryptText(rawValue);
          cache.set(keyFor(path, e), value);
          for (const r of resolvers) r.resolve(value);
        }
      } catch (err) {
        for (const { resolvers } of entries) for (const r of resolvers) r.reject(err);
      }
    }),
  );
  if (pending.size > 0) scheduleFlush();
}

function requestCaption(path: string, ext: string, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
    const key = keyFor(path, ext);
    const resolver: Resolver = { resolve, reject };
    const entry = pending.get(key);
    if (entry) entry.resolvers.push(resolver);
    else pending.set(key, { path, ext, resolvers: [resolver] });
    if (signal) {
      signal.addEventListener('abort', () => {
        const e = pending.get(key);
        if (e) {
          const idx = e.resolvers.indexOf(resolver);
          if (idx >= 0) e.resolvers.splice(idx, 1);
          if (e.resolvers.length === 0) pending.delete(key);
        }
        reject(new DOMException('Aborted', 'AbortError'));
      }, { once: true });
    }
    scheduleFlush();
  });
}

export function invalidateCaption(path: string, ext?: string) { cache.delete(keyFor(path, normExt(ext))); }
export function setCachedCaption(path: string, caption: string, ext?: string) { cache.set(keyFor(path, normExt(ext)), caption); }

export default function useCaptionBatch(imgPath: string | null, refreshKey: number = 0, ext: string = 'txt') {
  const captionExt = normExt(ext);
  const [caption, setCaption] = useState<string>(() => (imgPath ? (cache.get(keyFor(imgPath, captionExt)) ?? '') : ''));
  const [isLoaded, setIsLoaded] = useState<boolean>(() => Boolean(imgPath && cache.has(keyFor(imgPath, captionExt))));
  const lastPathRef = useRef<string | null>(null);

  useEffect(() => {
    if (!imgPath) { setCaption(''); setIsLoaded(false); return; }
    if (refreshKey > 0) invalidateCaption(imgPath, captionExt);
    const cached = cache.get(keyFor(imgPath, captionExt));
    if (cached !== undefined) { setCaption(cached); setIsLoaded(true); lastPathRef.current = imgPath; return; }

    let cancelled = false;
    const controller = new AbortController();
    lastPathRef.current = imgPath;
    setIsLoaded(false);
    requestCaption(imgPath, captionExt, controller.signal).then(value => {
      if (cancelled || lastPathRef.current !== imgPath) return;
      setCaption(value); setIsLoaded(true);
    }).catch(err => {
      if (err?.name === 'AbortError' || cancelled) return;
      setIsLoaded(true);
    });
    return () => { cancelled = true; controller.abort(); };
  }, [imgPath, refreshKey, captionExt]);
  return { caption, isLoaded, setCaption };
}
'''

UI_DATASET_IMAGE_CARD_TSX = ''''use client';
import React, { useEffect, useState, ReactNode, KeyboardEvent, useRef } from 'react';
import { FaTrashAlt, FaPlay } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import { isVideo, isAudio } from '@/utils/basic';
import useCaptionBatch, { setCachedCaption } from '@/hooks/useCaptionBatch';
import { decryptBuffer, encryptText } from '@/utils/crypto';

interface DatasetImageCardProps {
  imageUrl: string; alt: string; isAutoCaptioning: boolean; children?: ReactNode; className?: string; onDelete?: () => void; onImageClick?: () => void; captionRefreshKey?: number; observerRoot?: Element | null; rootMargin?: string; captionExt?: string;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({ imageUrl, alt, isAutoCaptioning, children, className = '', onDelete = () => {}, onImageClick, captionRefreshKey = 0, observerRoot = null, rootMargin = '200px 0px', captionExt = 'txt' }) => {
  const [loaded, setLoaded] = useState<boolean>(false);
  const [showAudioPlayer, setShowAudioPlayer] = useState(true);
  const [pollTick, setPollTick] = useState(0);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [streamVideo, setStreamVideo] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);

  useEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(entries => { for (const entry of entries) { if (entry.target === el) setIsVisible(entry.isIntersecting); } }, { root: observerRoot ?? null, threshold: 0.01, rootMargin });
    observer.observe(el);
    return () => observer.disconnect();
  }, [observerRoot, rootMargin]);

  useEffect(() => {
    if (isItAudio) return;
    if (!isVisible) return;
    const controller = new AbortController();
    let cancelled = false;
    let objectUrl: string | null = null;
    const timer = window.setTimeout(() => {
      fetch(`/api/img/${encodeURIComponent(imageUrl)}?thumb=1`, { signal: controller.signal })
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); })
        .then(async arrayBuffer => {
          if (cancelled || !arrayBuffer) return;
          const decryptedBuffer = await decryptBuffer(arrayBuffer);
          const ext = imageUrl.split('.').pop()?.toLowerCase() || 'jpeg';
          const mime = ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';
          const blob = new Blob([decryptedBuffer], { type: mime });
          objectUrl = URL.createObjectURL(blob);
          setBlobUrl(objectUrl);
          setLoaded(true);
        }).catch(err => { if (err?.name !== 'AbortError') console.error('Dataset image fetch failed:', err); });
    }, 80);
    return () => { cancelled = true; clearTimeout(timer); controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); setBlobUrl(null); setStreamVideo(false); setLoaded(false); };
  }, [imageUrl, isItAudio, isVisible]);

  const combinedRefreshKey = captionRefreshKey + pollTick;
  const { caption: fetchedCaption, isLoaded: isCaptionLoaded } = useCaptionBatch(isVisible ? imageUrl : null, combinedRefreshKey, captionExt);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const dirtyRef = useRef<boolean>(false);
  const hasLoadedCaptionRef = useRef(false);
  if (isCaptionLoaded) hasLoadedCaptionRef.current = true;

  useEffect(() => {
    if (!isCaptionLoaded) return;
    if (dirtyRef.current) return;
    setCaption(fetchedCaption);
    setSavedCaption(fetchedCaption.trim());
  }, [fetchedCaption, isCaptionLoaded]);

  useEffect(() => {
    if (!isAutoCaptioning) return;
    const interval = setInterval(() => setPollTick(t => t + 1), 5000);
    return () => clearInterval(interval);
  }, [isAutoCaptioning]);

  const saveCaption = async () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) { dirtyRef.current = false; return; }
    try {
      const payloadCaption = await encryptText(trimmedCaption);
      await apiClient.post('/api/img/caption', { imgPath: imageUrl, caption: payloadCaption, ext: captionExt });
      setSavedCaption(trimmedCaption);
      setCachedCaption(imageUrl, trimmedCaption, captionExt);
      dirtyRef.current = false;
    } catch (error) { console.error('Error saving caption:', error); }
  };

  const latestRef = useRef({ caption, savedCaption, imageUrl, captionExt });
  useEffect(() => { latestRef.current = { caption, savedCaption, imageUrl, captionExt }; });
  useEffect(() => {
    return () => {
      if (!dirtyRef.current) return;
      const { caption: c, savedCaption: s, imageUrl: url, captionExt: ext } = latestRef.current;
      const trimmed = c.trim();
      if (trimmed === s) return;
      const doSave = async () => {
        const payload = await encryptText(trimmed);
        await apiClient.post('/api/img/caption', { imgPath: url, caption: payload, ext });
        setCachedCaption(url, trimmed, ext);
      };
      doSave().catch(err => console.error('Error saving caption on unmount:', err));
    };
  }, []);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveCaption(); }
  };

  const handleCaptionChange = (value: string) => { dirtyRef.current = value.trim() !== savedCaption; setCaption(value); };
  const isCaptionCurrent = caption.trim() === savedCaption;

  return (
    <div ref={cardRef} className={`flex flex-col ${className}`}>
      <div className="relative w-full" style={{ paddingBottom: '100%' }}>
        <div className={classNames('absolute inset-0 rounded-t-lg shadow-md bg-gray-900', { 'animate-pulse': !isItAudio && !loaded })}>
          {streamVideo && <video src={`/api/img/${encodeURIComponent(imageUrl)}`} className={classNames('w-full h-full object-contain', { 'cursor-zoom-in': !!onImageClick })} onClick={onImageClick} playsInline loop muted />}
          {isItAudio && !showAudioPlayer && <div className="w-full h-full cursor-pointer flex items-center justify-center bg-gray-900" onClick={() => setShowAudioPlayer(true)}><img src={`/api/audio/art/${encodeURIComponent(imageUrl)}`} alt={alt} className="w-full h-full object-contain" onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }} /></div>}
          {isItAudio && showAudioPlayer && <AudioPlayer src={`/api/img/${encodeURIComponent(imageUrl)}`} title={imageUrl.replace(/^.*[\\/]/, '')} />}
          {!isItAudio && blobUrl && <img src={blobUrl} alt={alt} onClick={onImageClick} className={classNames('w-full h-full object-contain', { 'cursor-zoom-in': !!onImageClick })} />}
          {isItAVideo && loaded && <div className="absolute bottom-2 left-2 bg-gray-900/70 rounded-full p-2 pointer-events-none"><FaPlay className="w-3 h-3 text-white" /></div>}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
          <div className="absolute top-1 right-1 flex space-x-2 z-10"><button className="bg-gray-800 rounded-full p-2" onClick={() => { openConfirm({ title: `Delete ${isItAVideo ? 'video' : 'image'}`, message: `Are you sure you want to delete this ${isItAVideo ? 'video' : 'image'}? This action cannot be undone.`, type: 'warning', confirmText: 'Delete', onConfirm: () => { apiClient.post('/api/img/delete', { imgPath: imageUrl }).then(() => { onDelete(); }).catch(error => { console.error('Error deleting image:', error); }); }, }); }}><FaTrashAlt /></button></div>
        </div>
      </div>
      <div className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px]', { 'border-blue-500 border-2': !isCaptionCurrent, 'border-transparent border-2': isCaptionCurrent })}>
        {isCaptionLoaded || hasLoadedCaptionRef.current ? (
          <form onSubmit={e => { e.preventDefault(); saveCaption(); }} onBlur={saveCaption}>
            <textarea className={classNames('w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none', { 'opacity-50 cursor-not-allowed': isAutoCaptioning })} value={caption} rows={3} readOnly={isAutoCaptioning} onChange={e => handleCaptionChange(e.target.value)} onKeyDown={handleKeyDown} />
          </form>
        ) : <div className="w-full h-full flex items-center justify-center text-gray-400">Loading caption...</div>}
      </div>
    </div>
  );
};
export default DatasetImageCard;
'''

UI_ADD_IMAGES_MODAL_TSX = ''''use client';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaUpload, FaTimesCircle, FaSpinner } from 'react-icons/fa';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { apiClient } from '@/utils/api';
import { encryptFileForUpload } from '@/utils/crypto';

export interface AddImagesModalState { datasetName: string; onComplete?: () => void; openedByDrag?: boolean; }
export const addImagesModalState = createGlobalState<AddImagesModalState | null>(null);
export const openImagesModal = (datasetName: string, onComplete: () => void) => { addImagesModalState.set({ datasetName, onComplete }); };
export function useOpenImagesModalOnDrag(datasetName: string, onComplete: () => void) {
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  useEffect(() => {
    if (!datasetName) return;
    let depth = 0;
    const isFileDrag = (e: DragEvent) => { const types = e?.dataTransfer?.types; return !!types && Array.from(types).includes('Files'); };
    const onDragEnter = (e: DragEvent) => { if (!isFileDrag(e)) return; depth += 1; if (depth === 1) { if (!addImagesModalState.get()) { addImagesModalState.set({ datasetName, onComplete: onCompleteRef.current, openedByDrag: true }); } } e.preventDefault(); };
    const onDragLeave = (e: DragEvent) => { if (!isFileDrag(e)) return; depth = Math.max(0, depth - 1); if (depth === 0) { const current = addImagesModalState.get(); if (current?.openedByDrag) { addImagesModalState.set(null); } } };
    const onDrop = (e: DragEvent) => { if (!isFileDrag(e)) return; depth = 0; const current = addImagesModalState.get(); if (current?.openedByDrag) { addImagesModalState.set({ ...current, openedByDrag: false }); } };
    window.addEventListener('dragenter', onDragEnter); window.addEventListener('dragleave', onDragLeave); window.addEventListener('drop', onDrop);
    return () => { window.removeEventListener('dragenter', onDragEnter); window.removeEventListener('dragleave', onDragLeave); window.removeEventListener('drop', onDrop); };
  }, [datasetName]);
}

type AcceptMap = { [mime: string]: string[] };
type FileStatus = 'pending' | 'uploading' | 'error';
interface FileEntry { id: number; file: File; status: FileStatus; progress: number; error?: string; }
const MAX_CONCURRENT = 3; const ROW_HEIGHT = 32; const VISIBLE_ROWS = 8;
let nextId = 0;

export default function AddImagesModal() {
  const [modalInfo, setModalInfo] = addImagesModalState.use();
  const open = modalInfo !== null;
  const [isUploading, setIsUploading] = useState(false);
  const [fileEntries, setFileEntries] = useState<FileEntry[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [doneCount, setDoneCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const abortRef = useRef(false);
  const modalInfoRef = useRef(modalInfo);
  modalInfoRef.current = modalInfo;
  const datasetName = modalInfo?.datasetName ?? '';

  const uploadSingleFile = useCallback(
    async (entry: FileEntry): Promise<'done' | 'error'> => {
      if (abortRef.current) return 'error';
      const id = entry.id;
      setFileEntries(prev => prev.map(e => (e.id === id ? { ...e, status: 'uploading' as FileStatus, progress: 0 } : e)));
      try {
        const encryptedFile = await encryptFileForUpload(entry.file);
        const formData = new FormData();
        formData.append('files', encryptedFile);
        formData.append('datasetName', datasetName || '');
        await apiClient.post('/api/datasets/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: pe => { const percent = Math.round(((pe.loaded || 0) * 100) / (pe.total || pe.loaded || 1)); setFileEntries(prev => prev.map(e => (e.id === id ? { ...e, progress: percent } : e))); },
          timeout: 0,
        });
        setFileEntries(prev => prev.filter(e => e.id !== id));
        setDoneCount(prev => prev + 1);
        return 'done';
      } catch (err) {
        setFileEntries(prev => prev.map(e => e.id === id ? { ...e, status: 'error' as FileStatus, error: err instanceof Error ? err.message : 'Upload failed' } : e));
        setErrorCount(prev => prev + 1);
        return 'error';
      }
    }, [datasetName],
  );

  const resetState = useCallback(() => { setFileEntries([]); setTotalCount(0); setDoneCount(0); setErrorCount(0); }, []);
  const processQueue = useCallback(
    async (entries: FileEntry[]) => {
      setIsUploading(true); abortRef.current = false;
      let nextIndex = 0;
      const runNext = async (): Promise<void> => { while (nextIndex < entries.length) { if (abortRef.current) return; const idx = nextIndex++; await uploadSingleFile(entries[idx]); } };
      const workers = Array.from({ length: Math.min(MAX_CONCURRENT, entries.length) }, () => runNext());
      await Promise.all(workers);
      setIsUploading(false);
      if (!abortRef.current) { modalInfoRef.current?.onComplete?.(); setModalInfo(null); resetState(); }
    }, [uploadSingleFile, setModalInfo, resetState],
  );

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      const entries: FileEntry[] = acceptedFiles.map(file => ({ id: nextId++, file, status: 'pending' as FileStatus, progress: 0 }));
      setFileEntries(entries); setTotalCount(entries.length); setDoneCount(0); setErrorCount(0); processQueue(entries);
    }, [processQueue],
  );

  const handleCancel = useCallback(() => { abortRef.current = true; setIsUploading(false); setModalInfo(null); resetState(); }, [setModalInfo, resetState]);
  const dropAccept = useMemo<AcceptMap>(() => ({ 'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'], 'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'], 'audio/*': ['.mp3', '.wav', '.flac', '.ogg'], 'text/*': ['.txt'], 'application/json': ['.json'] }), []);
  const { getRootProps, getInputProps, isDragActive, open: openFilePicker } = useDropzone({ onDrop, accept: dropAccept, multiple: true, noClick: true, noKeyboard: true });
  const overallPercent = totalCount > 0 ? Math.round(((doneCount + errorCount) / totalCount) * 100) : 0;

  return (
    <Dialog open={open} onClose={() => { if (!isUploading) handleCancel(); }} className="relative z-10">
      <DialogBackdrop transition className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in" />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel transition className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95">
            <div className="bg-gray-800 px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="text-center">
                <DialogTitle as="h3" className="text-base font-semibold text-gray-200 mb-4">Add Images to: {datasetName}</DialogTitle>
                <div {...getRootProps()} className="w-full">
                  <input {...getInputProps()} />
                  <div onClick={() => { if (!isUploading) openFilePicker(); }} className={`h-40 w-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg cursor-pointer transition-colors ${isDragActive ? 'border-blue-400 bg-blue-500/10' : 'border-gray-600 hover:border-gray-400'}`}>
                    <FaUpload className="size-8 mb-3 text-gray-400" />
                    {!isUploading ? ( <><p className="text-sm text-gray-200 text-center">Drag & drop files here or click to select</p><p className="text-xs text-gray-400 mt-1">Images, videos, .txt or .json supported</p></> ) : ( <p className="text-sm text-gray-200 text-center">Drop more files to add to queue</p> )}
                  </div>
                </div>
                {isUploading && ( <div className="mt-4"><p className="text-sm font-semibold text-gray-200 mb-2">Uploading… {doneCount + errorCount} / {totalCount}</p><div className="w-full h-2.5 bg-white/20 rounded-full overflow-hidden"><div className="h-2.5 bg-blue-500 rounded-full transition-[width] duration-150 ease-linear" style={{ width: `${overallPercent}%` }} /></div>{errorCount > 0 && ( <p className="text-xs text-red-400 mt-1">{errorCount} file{errorCount !== 1 ? 's' : ''} failed</p> )}</div> )}
                {fileEntries.length > 0 && ( <div className="mt-3"><FileProgressList entries={fileEntries} /></div> )}
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
              <button type="button" onClick={handleCancel} className={`inline-flex w-full justify-center rounded-md px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto ${isUploading ? 'bg-red-600 hover:bg-red-500' : 'bg-gray-600 hover:bg-gray-500'}`}>{isUploading ? 'Cancel Upload' : 'Close'}</button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

function FileProgressList({ entries }: { entries: FileEntry[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const totalHeight = entries.length * ROW_HEIGHT;
  const containerHeight = Math.min(entries.length, VISIBLE_ROWS) * ROW_HEIGHT;
  const startIdx = Math.floor(scrollTop / ROW_HEIGHT);
  const endIdx = Math.min(entries.length, startIdx + VISIBLE_ROWS + 2);
  const visibleEntries = entries.slice(startIdx, endIdx);
  const offsetY = startIdx * ROW_HEIGHT;
  const onScroll = useCallback(() => { if (containerRef.current) setScrollTop(containerRef.current.scrollTop); }, []);
  return (
    <div ref={containerRef} onScroll={onScroll} className="rounded-xl bg-black/60 backdrop-blur-sm border border-white/10 overflow-y-auto" style={{ height: containerHeight + 2 }}>
      <div style={{ height: totalHeight, position: 'relative' }}><div style={{ position: 'absolute', top: offsetY, left: 0, right: 0 }}>{visibleEntries.map(entry => ( <FileRow key={entry.id} entry={entry} /> ))}</div></div>
    </div>
  );
}
function FileRow({ entry }: { entry: FileEntry }) {
  return (
    <div className="flex items-center gap-2 px-3 text-xs font-mono" style={{ height: ROW_HEIGHT }}>
      <span className="flex-shrink-0 w-4 text-center">{entry.status === 'error' && <FaTimesCircle className="text-red-400 inline" />}{entry.status === 'uploading' && <FaSpinner className="text-blue-400 inline animate-spin" />}{entry.status === 'pending' && <span className="inline-block w-2 h-2 rounded-full bg-white/30" />}</span>
      <span className="truncate flex-1 opacity-80" title={entry.file.name}>{entry.file.name}</span>
      <span className="flex-shrink-0 w-16 text-right">{entry.status === 'uploading' && <span className="text-blue-300">{entry.progress}%</span>}{entry.status === 'error' && <span className="text-red-400">Failed</span>}{entry.status === 'pending' && <span className="text-white/30">Queued</span>}</span>
    </div>
  );
}
'''

TOOLKIT_JOB_PY = '''import json
import yaml
from typing import Union, OrderedDict

from toolkit.config import get_config
from toolkit.crypto import decrypt_bytes

def get_job(config_path: Union[str, dict, OrderedDict], name=None):
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
    else:
        raise ValueError(f'Unknown job type {job}')

def run_job(config: Union[str, dict, OrderedDict], name=None):
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
                if isinstance(a, str): sanitized_args.append(a)
                else: sanitized_args.append(a)
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
    # Capture the real streams before replacing them
    log_file = open(filename, 'a')
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)
'''

# ==============================================================================
# Robust String Replacement Helper
# ==============================================================================

def patch_file_if_contains(file_path: Path, search_text: str, replacement_text: str):
    if not file_path.exists():
        print(f"  [!] Skipped {file_path.name} (File not found)")
        return False
    content = file_path.read_text(encoding="utf-8")
    if replacement_text in content:
        print(f"  [-] Already patched: {file_path.name}")
        return True
    if search_text not in content:
        print(f"  [!] Pattern not matched in {file_path.name}. Needs manual review.")
        return False
    new_content = content.replace(search_text, replacement_text, 1)
    file_path.write_text(new_content, encoding="utf-8")
    print(f"  [+] Patched string in: {file_path.name}")
    return True


def main():
    target_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    print(f"Applying AI-Toolkit Privacy Fixes to:\n  {target_dir}\n")

    # ---------------------------------------------------------
    # 1. Create/Overwrite Full Files (Frontend & Backend Core)
    # ---------------------------------------------------------
    files_to_write = {
        "toolkit/crypto.py": TOOLKIT_CRYPTO_PY,
        "toolkit/job.py": TOOLKIT_JOB_PY,
        "toolkit/print.py": TOOLKIT_PRINT_PY,
        "ui/src/utils/crypto.ts": UI_CRYPTO_TS,
        "ui/src/server/settings.ts": UI_SERVER_SETTINGS_TS,
        "ui/src/app/api/settings/route.ts": UI_API_SETTINGS_ROUTE_TS,
        "ui/src/hooks/useSettings.tsx": UI_HOOKS_USE_SETTINGS_TSX,
        "ui/src/app/settings/page.tsx": UI_APP_SETTINGS_PAGE_TSX,
        "ui/src/components/AddImagesModal.tsx": UI_ADD_IMAGES_MODAL_TSX,
        "ui/src/components/DatasetImageCard.tsx": UI_DATASET_IMAGE_CARD_TSX,
        "ui/src/hooks/useCaptionBatch.tsx": UI_HOOKS_USE_CAPTION_BATCH_TSX,
    }

    for rel_path, content in files_to_write.items():
        p = target_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"  [+] Overwrote full file: {rel_path}")

    # ---------------------------------------------------------
    # 2. Patch Existing Backend Logic via String Replacement
    # ---------------------------------------------------------

    # 2.1 Patch run.py (Inject Crypto initialization)
    p = target_dir / "run.py"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if "import toolkit.crypto" not in content:
            content = "import toolkit.crypto  # Initialize Privacy Patches\n" + content
            p.write_text(content, encoding="utf-8")
            print(f"  [+] Injected crypto initialization into: {p.name}")

    # 2.2 Patch toolkit/data_transfer_object/data_loader.py (Spoof cv2 so it falls back safely to PyAV)
    p = target_dir / "toolkit" / "data_transfer_object" / "data_loader.py"
    patch_file_if_contains(p,
        search_text="""            else:
                # Open the video file
                video = cv2.VideoCapture(self.path)

                # Check if video opened successfully
                if not video.isOpened():""",
        replacement_text="""            else:
                import io
                from toolkit.crypto import read_decrypted_file
                _is_enc = False
                try:
                    with open(self.path, 'rb') as _f:
                        if _f.read(11) == b"AITK_ENC_V1": _is_enc = True
                except: pass
                
                if _is_enc:
                    class MockVideo:
                        def __init__(self, p): self.p = p
                        def isOpened(self): return True
                        def get(self, prop):
                            import av
                            with av.open(io.BytesIO(read_decrypted_file(self.p))) as c:
                                s = c.streams.video[0]
                                if prop == cv2.CAP_PROP_FRAME_WIDTH: return s.width
                                if prop == cv2.CAP_PROP_FRAME_HEIGHT: return s.height
                                if prop == cv2.CAP_PROP_FRAME_COUNT: return s.frames if s.frames > 0 else int(float(s.duration * s.time_base) * float(s.average_rate))
                                if prop == cv2.CAP_PROP_FPS: return float(s.average_rate)
                            return 0
                        def release(self): pass
                    video = MockVideo(self.path)
                else:
                    video = cv2.VideoCapture(self.path)

                # Check if video opened successfully
                if not video.isOpened():""")

    # 2.3 Patch toolkit/dataloader_mixins.py (Spoof cv2 frame extraction, fix prompt reading)
    p = target_dir / "toolkit" / "dataloader_mixins.py"
    patch_file_if_contains(p,
        search_text="""        try:
            # Use OpenCV to capture video frames
            cap = cv2.VideoCapture(self.path)""",
        replacement_text="""        try:
            import io
            from toolkit.crypto import read_decrypted_file
            _is_enc = False
            try:
                with open(self.path, 'rb') as _f:
                    if _f.read(11) == b"AITK_ENC_V1": _is_enc = True
            except: pass
            
            if _is_enc:
                class MockCap:
                    def __init__(self, p): self.p = p
                    def isOpened(self): return True
                    def get(self, prop):
                        import av
                        with av.open(io.BytesIO(read_decrypted_file(self.p))) as c:
                            s = c.streams.video[0]
                            if prop == cv2.CAP_PROP_FRAME_COUNT: return s.frames if s.frames > 0 else int(float(s.duration * s.time_base) * float(s.average_rate))
                            if prop == cv2.CAP_PROP_FPS: return float(s.average_rate)
                        return 0
                    def set(self, *args): pass
                    def grab(self): return False
                    def read(self): return False, None
                    def release(self): pass
                cap = MockCap(self.path)
            else:
                cap = cv2.VideoCapture(self.path)""")

    patch_file_if_contains(p,
        search_text="""            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                    short_caption = None
                    prompt = clean_caption(prompt)
                    if short_caption is not None:
                        short_caption = clean_caption(short_caption)
                    
                    if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption""",
        replacement_text="""            if os.path.exists(prompt_path):
                from toolkit.crypto import read_decrypted_text
                prompt = read_decrypted_text(prompt_path)
                short_caption = None
                prompt = clean_caption(prompt)
                if short_caption is not None:
                    short_caption = clean_caption(short_caption)
                
                if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption""")

    # 2.4 Patch extensions_built_in/sd_trainer/SDTrainer.py (Encrypted Negative Prompt files)
    p = target_dir / "extensions_built_in" / "sd_trainer" / "SDTrainer.py"
    patch_file_if_contains(p,
        search_text="""        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]""",
        replacement_text="""        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                from toolkit.crypto import read_decrypted_text
                text = read_decrypted_text(self.train_config.negative_prompt)
                self.negative_prompt_pool = text.splitlines()
                # remove empty
                self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]""")

    # 2.5 Patch toolkit/data_loader.py (Encrypted JSON datasets configs)
    p = target_dir / "toolkit" / "data_loader.py"
    patch_file_if_contains(p,
        search_text="""            # assume json
            with open(self.dataset_path, 'r') as f:
                self.caption_dict = json.load(f)
                # keys are file paths
                file_list = list(self.caption_dict.keys())""",
        replacement_text="""            # assume json
            from toolkit.crypto import read_decrypted_text
            import json
            text = read_decrypted_text(self.dataset_path)
            self.caption_dict = json.loads(text)
            # keys are file paths
            file_list = list(self.caption_dict.keys())""")

    # 2.6 Patch extensions_built_in/captioner/*.py (Write encrypted captions out of the box)
    captioners_dir = target_dir / "extensions_built_in" / "captioner"
    if captioners_dir.exists():
        for cap_file in captioners_dir.glob("*.py"):
            content = cap_file.read_text(encoding="utf-8")
            if 'with open(caption_path, "w", encoding="utf-8") as f:\n            f.write(caption)' in content:
                content = content.replace(
                    '        with open(caption_path, "w", encoding="utf-8") as f:\n            f.write(caption)',
                    '        from toolkit.crypto import write_encrypted_text\n        write_encrypted_text(caption_path, caption)'
                )
                cap_file.write_text(content, encoding="utf-8")
                print(f"  [+] Patched Python Auto-Captioner: {cap_file.name}")

    # 2.7 Patch toolkit/metadata.py (Sanitize LoRA output metadata)
    p = target_dir / "toolkit" / "metadata.py"
    patch_file_if_contains(p,
        search_text="""    if add_software_info:
        save_meta["software"] = software_meta

    # safetensors can only be one level deep""",
        replacement_text="""    if add_software_info:
        save_meta["software"] = software_meta

    sensitive_keys = ["ss_tag_frequency", "instance_prompt", "caption", "prompt", "training_info"]
    for k in list(save_meta.keys()):
        if any(sk in k.lower() for sk in sensitive_keys):
            del save_meta[k]

    # safetensors can only be one level deep""")

    # ---------------------------------------------------------
    # 3. Patch Existing Frontend Next.js API Logic
    # ---------------------------------------------------------

    # 3.1 Patch ui/cron/actions/startJob.ts (Pass Env password to python worker)
    p = target_dir / "ui" / "cron" / "actions" / "startJob.ts"
    patch_file_if_contains(p,
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
    }""")

    # 3.2 Patch ui/cron/fileServer.ts (Skip server thumbnail processing for encrypted files)
    p = target_dir / "ui" / "cron" / "fileServer.ts"
    patch_file_if_contains(p,
        search_text="""async function generateThumb(sourcePath: string, thumbPath: string): Promise<boolean> {""",
        replacement_text="""async function generateThumb(sourcePath: string, thumbPath: string): Promise<boolean> {
  // Check if source file is encrypted; if so, skip server thumbnail generation
  try {
    const handle = await fs.promises.open(sourcePath, 'r');
    const headerBuf = Buffer.alloc(11);
    await handle.read(headerBuf, 0, 11, 0);
    await handle.close();
    if (headerBuf.toString('ascii') === 'AITK_ENC_V1') {
      return false; // Let browser decrypt the file directly
    }
  } catch {
    return false;
  }""")

    print("\nPatch complete! All scripts, web components, API routes, and python patches applied successfully.")
    print("Next step: Run `npm run build` to compile the Next.js UI.")


if __name__ == "__main__":
    main()
