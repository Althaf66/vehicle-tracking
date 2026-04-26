import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Navbar.css';

const Navbar = () => {
  const { user, logout, isAdmin } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActive = (path) => {
    return location.pathname === path ? 'active' : '';
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          Vehicle Tracking System
        </Link>

        <div className="navbar-menu">
          <Link to="/" className={`navbar-link ${isActive('/')}`}>
            Home
          </Link>

          <Link
            to="/vehicle-log"
            className={`navbar-link ${isActive('/vehicle-log')}`}
          >
            Vehicle Log
          </Link>

          {isAdmin() && (
            <Link
              to="/authorized-vehicles"
              className={`navbar-link ${isActive('/authorized-vehicles')}`}
            >
              Authorized Vehicles
            </Link>
          )}

          <div className="navbar-user">
            <span className="user-info">
              {user?.username} ({user?.role})
            </span>
            <button onClick={handleLogout} className="btn btn-secondary btn-sm">
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
