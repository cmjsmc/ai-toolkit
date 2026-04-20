import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';
import { encryptPayload, decryptPayload } from './crypto';

export const isAuthorizedState = createGlobalState(false);

export const apiClient = axios.create();

// Add a request interceptor to add token from localStorage and encrypt JSON payloads
apiClient.interceptors.request.use(async config => {
  const token = localStorage.getItem('AI_TOOLKIT_AUTH');
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
    
    // Encrypt JSON payloads (ignore FormData which handles media files)
    if (config.data && !(config.data instanceof FormData)) {
      config.data = await encryptPayload(config.data, token);
    }
  }
  return config;
});

// Add a response interceptor to handle 401 errors and automatically decrypt responses
apiClient.interceptors.response.use(
  async response => {
    const token = localStorage.getItem('AI_TOOLKIT_AUTH');
    
    // Transparently decrypt the encrypted envelope back to plaintext JSON
    if (token && response.data && typeof response.data === 'object' && 'encryptedPayload' in response.data) {
      try {
        response.data = await decryptPayload(response.data.encryptedPayload, token);
      } catch (error) {
        console.error('Decryption error:', error);
      }
    }
    return response; // Return successful responses
  },
  error => {
    // Check if the error is a 401 Unauthorized
    if (error.response && error.response.status === 401) {
      // Clear the auth token from localStorage
      localStorage.removeItem('AI_TOOLKIT_AUTH');
      isAuthorizedState.set(false);
    }

    // Reject the promise with the error so calling code can still catch it
    return Promise.reject(error);
  },
);
