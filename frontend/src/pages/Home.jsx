import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import UnauthorizedVehicleTracker from '../components/UnauthorizedVehicleTracker';
import VehicleDetectionLog from '../components/VehicleDetectionLog';
import { camerasAPI } from '../services/api';
import './Home.css';

const Home = () => {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchCameras();
  }, []);

  const fetchCameras = async () => {
    try {
      const response = await camerasAPI.getAll();
      setCameras(response.data.cameras);
      setError('');
    } catch (err) {
      setError('Failed to fetch cameras');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getCameraStreamUrl = (cameraId) => {
    const token = localStorage.getItem('token');
    return `http://localhost:8001/api/stream/${cameraId}?token=${token}`;
  };

  const getCameraTitle = (camera) => {
    if (camera.camera_type === 'entrance') {
      return `${camera.camera_name} 🔍`;
    }
    return camera.camera_name;
  };

  const getCameraDescription = (camera) => {
    if (camera.camera_type === 'entrance') {
      return 'License Plate Recognition • Authorization Check';
    }
    return 'Vehicle Detection • Tracking';
  };

  if (loading) {
    return (
      <Layout>
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="alert alert-error">{error}</div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="home-container">
        <h1 className="page-title">Multi-Camera Vehicle Tracking Dashboard</h1>

        {/* Camera Grid - 2x2 layout */}
        <div className="camera-grid">
          {cameras.map((camera) => (
            <div key={camera.camera_id} className="camera-card">
              <div className="camera-header">
                <h3 className="camera-title">{getCameraTitle(camera)}</h3>
                <span className={`camera-status ${camera.status}`}>
                  {camera.status === 'online' ? '🟢' : '🔴'} {camera.status}
                </span>
              </div>
              <div className="camera-description">{getCameraDescription(camera)}</div>

              {camera.status === 'online' ? (
                <div className="camera-stream">
                  <img
                    src={getCameraStreamUrl(camera.camera_id)}
                    alt={camera.camera_name}
                    className="stream-image"
                  />
                </div>
              ) : (
                <div className="camera-offline">
                  <div className="offline-message">
                    <span className="offline-icon">📹</span>
                    <p>Camera Offline</p>
                    <small>Video file not found</small>
                  </div>
                </div>
              )}

              {camera.camera_type === 'entrance' && (
                <div className="camera-legend">
                  <div className="legend-item">
                    <span className="legend-color green"></span>
                    <span>Authorized</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-color red"></span>
                    <span>Unauthorized</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-color yellow"></span>
                    <span>No Plate</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Vehicle Detection Log */}
        <div className="detection-log-section">
          <VehicleDetectionLog />
        </div>

        {/* Unauthorized Vehicle Tracking Table */}
        <div className="tracking-section">
          <UnauthorizedVehicleTracker />
        </div>
      </div>
    </Layout>
  );
};

export default Home;
