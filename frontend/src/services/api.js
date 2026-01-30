import axios from 'axios';

// Create axios instance with default config
const api = axios.create({
  baseURL: 'http://localhost:8001',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ========== AUTH API ==========

export const authAPI = {
  signup: (userData) => api.post('/api/auth/signup', userData),
  login: (credentials) => api.post('/api/auth/login', credentials),
  getMe: () => api.get('/api/auth/me'),
};

// ========== AUTHORIZED VEHICLES API ==========

export const vehiclesAPI = {
  getAll: () => api.get('/api/vehicles'),
  getById: (id) => api.get(`/api/vehicles/${id}`),
  create: (vehicleData) => api.post('/api/vehicles', vehicleData),
  update: (id, vehicleData) => api.put(`/api/vehicles/${id}`, vehicleData),
  delete: (id) => api.delete(`/api/vehicles/${id}`),
};

// ========== UNAUTHORIZED TRACKING API ==========

export const trackingAPI = {
  getAll: (activeOnly = false) =>
    api.get('/api/tracking/unauthorized', { params: { active_only: activeOnly } }),
  getActive: () => api.get('/api/tracking/unauthorized/active'),
  getById: (trackingId) => api.get(`/api/tracking/unauthorized/${trackingId}`),
};

// ========== CAMERAS API ==========

export const camerasAPI = {
  getAll: () => api.get('/api/cameras'),
  getStreamUrl: (cameraId) => {
    const token = localStorage.getItem('token');
    return `http://localhost:8001/api/stream/${cameraId}?token=${token}`;
  },
};

// ========== STATISTICS API ==========

export const statsAPI = {
  get: () => api.get('/api/stats'),
};

export default api;
