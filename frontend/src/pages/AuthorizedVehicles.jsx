import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { vehiclesAPI } from '../services/api';
import './AuthorizedVehicles.css';

const AuthorizedVehicles = () => {
  const [vehicles, setVehicles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editingVehicle, setEditingVehicle] = useState(null);
  const [formData, setFormData] = useState({
    plate_number: '',
    vehicle_owner: '',
    vehicle_type: 'car',
    notes: '',
  });
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  useEffect(() => {
    fetchVehicles();
  }, []);

  const fetchVehicles = async () => {
    try {
      const response = await vehiclesAPI.getAll();
      setVehicles(response.data);
      setError('');
    } catch (err) {
      setError('Failed to fetch vehicles');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = () => {
    setEditingVehicle(null);
    setFormData({
      plate_number: '',
      vehicle_owner: '',
      vehicle_type: 'car',
      notes: '',
    });
    setShowModal(true);
  };

  const handleEdit = (vehicle) => {
    setEditingVehicle(vehicle);
    setFormData({
      plate_number: vehicle.plate_number,
      vehicle_owner: vehicle.vehicle_owner || '',
      vehicle_type: vehicle.vehicle_type || 'car',
      notes: vehicle.notes || '',
    });
    setShowModal(true);
  };

  const handleDelete = async (vehicle) => {
    setDeleteConfirm(vehicle);
  };

  const confirmDelete = async () => {
    try {
      await vehiclesAPI.delete(deleteConfirm.id);
      setVehicles(vehicles.filter((v) => v.id !== deleteConfirm.id));
      setDeleteConfirm(null);
      setError('');
    } catch (err) {
      setError('Failed to delete vehicle');
      console.error(err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      if (editingVehicle) {
        // Update existing vehicle
        await vehiclesAPI.update(editingVehicle.id, {
          vehicle_owner: formData.vehicle_owner,
          vehicle_type: formData.vehicle_type,
          notes: formData.notes,
        });
      } else {
        // Create new vehicle
        await vehiclesAPI.create(formData);
      }

      setShowModal(false);
      fetchVehicles();
    } catch (err) {
      setError(err.response?.data?.detail || 'Operation failed');
      console.error(err);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
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

  return (
    <Layout>
      <div className="authorized-vehicles-container">
        <div className="page-header">
          <h1 className="page-title">Authorized Vehicles Management</h1>
          <button onClick={handleAdd} className="btn btn-primary">
            + Add Vehicle
          </button>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <div className="card">
          {vehicles.length === 0 ? (
            <div className="no-data">
              No authorized vehicles. Add your first vehicle to get started.
            </div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Plate Number</th>
                  <th>Owner</th>
                  <th>Type</th>
                  <th>Notes</th>
                  <th>Added On</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {vehicles.map((vehicle) => (
                  <tr key={vehicle.id}>
                    <td className="plate-cell">{vehicle.plate_number}</td>
                    <td>{vehicle.vehicle_owner || '-'}</td>
                    <td>
                      <span className="vehicle-type-badge">{vehicle.vehicle_type || '-'}</span>
                    </td>
                    <td className="notes-cell">{vehicle.notes || '-'}</td>
                    <td>{new Date(vehicle.created_at).toLocaleDateString()}</td>
                    <td>
                      <div className="action-buttons">
                        <button
                          onClick={() => handleDelete(vehicle)}
                          className="btn btn-danger btn-sm"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Add/Edit Modal */}
        {showModal && (
          <div className="modal-overlay" onClick={() => setShowModal(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                {editingVehicle ? 'Edit Vehicle' : 'Add New Vehicle'}
              </div>

              <form onSubmit={handleSubmit}>
                <div className="form-group">
                  <label htmlFor="plate_number">Plate Number *</label>
                  <input
                    type="text"
                    id="plate_number"
                    name="plate_number"
                    className="form-control"
                    value={formData.plate_number}
                    onChange={handleChange}
                    required
                    disabled={!!editingVehicle}
                    placeholder="e.g., ABC1234"
                  />
                  {editingVehicle && (
                    <small className="form-text">Plate number cannot be changed</small>
                  )}
                </div>

                <div className="form-group">
                  <label htmlFor="vehicle_owner">Owner</label>
                  <input
                    type="text"
                    id="vehicle_owner"
                    name="vehicle_owner"
                    className="form-control"
                    value={formData.vehicle_owner}
                    onChange={handleChange}
                    placeholder="e.g., John Doe"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="vehicle_type">Vehicle Type</label>
                  <select
                    id="vehicle_type"
                    name="vehicle_type"
                    className="form-control"
                    value={formData.vehicle_type}
                    onChange={handleChange}
                  >
                    <option value="car">Car</option>
                    <option value="bus">Bus</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="notes">Notes</label>
                  <textarea
                    id="notes"
                    name="notes"
                    className="form-control"
                    value={formData.notes}
                    onChange={handleChange}
                    rows="3"
                    placeholder="Additional information..."
                  />
                </div>

                <div className="modal-footer">
                  <button type="button" onClick={() => setShowModal(false)} className="btn btn-secondary">
                    Cancel
                  </button>
                  <button type="submit" className="btn btn-primary">
                    {editingVehicle ? 'Update' : 'Add'} Vehicle
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Delete Confirmation Modal */}
        {deleteConfirm && (
          <div className="modal-overlay" onClick={() => setDeleteConfirm(null)}>
            <div className="modal-content delete-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">Confirm Deletion</div>
              <p>
                Are you sure you want to delete vehicle <strong>{deleteConfirm.plate_number}</strong>?
              </p>
              <p className="delete-warning">This action cannot be undone.</p>
              <div className="modal-footer">
                <button onClick={() => setDeleteConfirm(null)} className="btn btn-secondary">
                  Cancel
                </button>
                <button onClick={confirmDelete} className="btn btn-danger">
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default AuthorizedVehicles;
