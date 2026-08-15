"""
apply_privacy_patch.py

Applies the complete privacy-oriented end-to-end encryption modifications to an ai-toolkit installation.
"""

import os
import sys
from pathlib import Path

# ==============================================================================
# Full File Contents
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
        raise ValueError("Encrypted content detected, but no password is set.")

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
  } catch (e) {
    console.error('Failed to fetch encryption password:', e);
  }
  return '';
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
    { name: 'PBKDF2', salt, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
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
      } catch {
        return data;
      }
    } else {
      buf = new TextEncoder().encode(data).buffer;
    }
  } else {
    buf = data;
  }
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
  imageUrl: string;
  alt: string;
  isAutoCaptioning: boolean;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
  onImageClick?: () => void;
  captionRefreshKey?: number;
  observerRoot?: Element | null;
  rootMargin?: string;
  captionExt?: string;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  isAutoCaptioning,
  children,
  className = '',
  onDelete = () => {},
  onImageClick,
  captionRefreshKey = 0,
  observerRoot = null,
  rootMargin = '200px 0px',
  captionExt = 'txt',
}) => {
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
    const observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.target === el) {
            setIsVisible(entry.isIntersecting);
          }
        }
      },
      { root: observerRoot ?? null, threshold: 0.01, rootMargin }
    );
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
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.arrayBuffer();
        })
        .then(async arrayBuffer => {
          if (cancelled || !arrayBuffer) return;
          const decryptedBuffer = await decryptBuffer(arrayBuffer);
          const ext = imageUrl.split('.').pop()?.toLowerCase() || 'jpeg';
          const mime = ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';
          const blob = new Blob([decryptedBuffer], { type: mime });
          objectUrl = URL.createObjectURL(blob);
          setBlobUrl(objectUrl);
          setLoaded(true);
        })
        .catch(err => {
          if (err?.name !== 'AbortError') console.error('Dataset image fetch failed:', err);
        });
    }, 80);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setBlobUrl(null);
      setStreamVideo(false);
      setLoaded(false);
    };
  }, [imageUrl, isItAudio, isVisible]);

  const combinedRefreshKey = captionRefreshKey + pollTick;
  const { caption: fetchedCaption, isLoaded: isCaptionLoaded } = useCaptionBatch(
    isVisible ? imageUrl : null,
    combinedRefreshKey,
    captionExt,
  );

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
    if (trimmedCaption === savedCaption) {
      dirtyRef.current = false;
      return;
    }
    try {
      const payloadCaption = await encryptText(trimmedCaption);
      await apiClient.post('/api/img/caption', { imgPath: imageUrl, caption: payloadCaption, ext: captionExt });
      setSavedCaption(trimmedCaption);
      setCachedCaption(imageUrl, trimmedCaption, captionExt);
      dirtyRef.current = false;
    } catch (error) {
      console.error('Error saving caption:', error);
    }
  };

  const latestRef = useRef({ caption, savedCaption, imageUrl, captionExt });
  useEffect(() => {
    latestRef.current = { caption, savedCaption, imageUrl, captionExt };
  });

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
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const handleCaptionChange = (value: string) => {
    dirtyRef.current = value.trim() !== savedCaption;
    setCaption(value);
  };

  const isCaptionCurrent = caption.trim() === savedCaption;

  return (
    <div ref={cardRef} className={`flex flex-col ${className}`}>
      <div className="relative w-full" style={{ paddingBottom: '100%' }}>
        <div
          className={classNames('absolute inset-0 rounded-t-lg shadow-md bg-gray-900', {
            'animate-pulse': !isItAudio && !loaded,
          })}
        >
          {streamVideo && (
            <video
              src={`/api/img/${encodeURIComponent(imageUrl)}`}
              className={classNames('w-full h-full object-contain', {
                'cursor-zoom-in': !!onImageClick,
              })}
              onClick={onImageClick}
              autoPlay={false}
              preload="metadata"
              playsInline
              loop
              muted
            />
          )}
          {isItAudio && !showAudioPlayer && (
            <div
              className="w-full h-full cursor-pointer flex items-center justify-center bg-gray-900"
              onClick={() => setShowAudioPlayer(true)}
            >
              <img
                src={`/api/audio/art/${encodeURIComponent(imageUrl)}`}
                alt={alt}
                className="w-full h-full object-contain"
                onError={e => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>
          )}
          {isItAudio && showAudioPlayer && (
            <AudioPlayer src={`/api/img/${encodeURIComponent(imageUrl)}`} title={imageUrl.replace(/^.*[\\/]/, '')} />
          )}
          {!isItAudio && blobUrl && (
            <img
              src={blobUrl}
              alt={alt}
              onClick={onImageClick}
              className={classNames('w-full h-full object-contain', {
                'cursor-zoom-in': !!onImageClick,
              })}
            />
          )}
          {isItAVideo && loaded && (
            <div className="absolute bottom-2 left-2 bg-gray-900/70 rounded-full p-2 pointer-events-none">
              <FaPlay className="w-3 h-3 text-white" />
            </div>
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
          <div className="absolute top-1 right-1 flex space-x-2 z-10">
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => {
                openConfirm({
                  title: `Delete ${isItAVideo ? 'video' : 'image'}`,
                  message: `Are you sure you want to delete this ${isItAVideo ? 'video' : 'image'}? This action cannot be undone.`,
                  type: 'warning',
                  confirmText: 'Delete',
                  onConfirm: () => {
                    apiClient
                      .post('/api/img/delete', { imgPath: imageUrl })
                      .then(() => {
                        console.log('Image deleted:', imageUrl);
                        onDelete();
                      })
                      .catch(error => {
                        console.error('Error deleting image:', error);
                      });
                  },
                });
              }}
            >
              <FaTrashAlt />
            </button>
          </div>
        </div>
      </div>
      <div
        className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px]', {
          'border-blue-500 border-2': !isCaptionCurrent,
          'border-transparent border-2': isCaptionCurrent,
        })}
      >
        {isCaptionLoaded || hasLoadedCaptionRef.current ? (
          <form
            onSubmit={e => {
              e.preventDefault();
              saveCaption();
            }}
            onBlur={saveCaption}
          >
            <textarea
              className={classNames('w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none', {
                'opacity-50 cursor-not-allowed': isAutoCaptioning,
              })}
              value={caption}
              rows={3}
              readOnly={isAutoCaptioning}
              onChange={e => handleCaptionChange(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </form>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">Loading caption...</div>
        )}
      </div>
    </div>
  );
};

export default DatasetImageCard;
'''


def main():
    target_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    print(f"Applying AI-Toolkit Privacy Fixes to:\n  {target_dir}\n")

    # 1. toolkit/crypto.py
    p = target_dir / "toolkit" / "crypto.py"
    p.write_text(TOOLKIT_CRYPTO_PY, encoding="utf-8")
    print(f"  [+] Updated: {p.relative_to(target_dir)}")

    # 2. ui/src/utils/crypto.ts
    p = target_dir / "ui" / "src" / "utils" / "crypto.ts"
    p.write_text(UI_CRYPTO_TS, encoding="utf-8")
    print(f"  [+] Updated: {p.relative_to(target_dir)}")

    # 3. ui/src/components/DatasetImageCard.tsx (Full Replacement)
    p = target_dir / "ui" / "src" / "components" / "DatasetImageCard.tsx"
    p.write_text(UI_DATASET_IMAGE_CARD_TSX, encoding="utf-8")
    print(f"  [+] Fixed via Full Replacement: {p.relative_to(target_dir)}")

    # 4. Fix useCaptionBatch.tsx to await decryptText
    p = target_dir / "ui" / "src" / "hooks" / "useCaptionBatch.tsx"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if "const password = getStoredPassword();" in content:
            content = content.replace(
                "const password = getStoredPassword();\n        for (const { path, ext: e, resolvers } of entries) {\n          const rawValue = captions[path] ?? '';\n          const value = await decryptText(rawValue, password);",
                "for (const { path, ext: e, resolvers } of entries) {\n          const rawValue = captions[path] ?? '';\n          const value = await decryptText(rawValue);"
            )
            p.write_text(content, encoding="utf-8")
            print(f"  [+] Fixed String Replace: {p.name}")

    # 5. Fix AddImagesModal.tsx to avoid passing password directly (now uses async getter inside)
    p = target_dir / "ui" / "src" / "components" / "AddImagesModal.tsx"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if "const password = getStoredPassword();" in content:
            content = content.replace(
                "const password = getStoredPassword();\n        const encryptedFile = await encryptFileForUpload(entry.file, password);",
                "const encryptedFile = await encryptFileForUpload(entry.file);"
            )
            p.write_text(content, encoding="utf-8")
            print(f"  [+] Fixed String Replace: {p.name}")

    # 6. Fix toolkit/dataloader_mixins.py to use `read_decrypted_text` for Base64 decode
    p = target_dir / "toolkit" / "dataloader_mixins.py"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if "raw_bytes = read_decrypted_file(prompt_path)" in content:
            content = content.replace(
                "                raw_bytes = read_decrypted_file(prompt_path)\n                try:\n                    prompt = raw_bytes.decode('utf-8')\n                except Exception:\n                    prompt = ''",
                "                from toolkit.crypto import read_decrypted_text\n                prompt = read_decrypted_text(prompt_path)"
            )
            p.write_text(content, encoding="utf-8")
            print(f"  [+] Fixed Python Text Decoder: {p.name}")

    # 7. Fix Built-in Python Captioners (e.g., Qwen3OmniCaptioner) to encrypt when saving!
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
                print(f"  [+] Fixed Python Auto-Captioner: {cap_file.name}")

    print("\nPatch complete! Run `npm run build` to clear the Next.js cache.")


if __name__ == "__main__":
    main()
