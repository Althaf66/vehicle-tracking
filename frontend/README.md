# Vehicle Tracking System - Frontend

React frontend application for the vehicle tracking system.

## Features

- User authentication (Login/Signup)
- Multi-camera live video streaming (2x2 grid)
- Real-time vehicle detection visualization
- License plate recognition display (Camera 1)
- Unauthorized vehicle tracking table with auto-refresh
- Admin panel for authorized vehicle management
- Role-based access control

## Tech Stack

- React 18
- React Router v6
- Axios for API calls
- TanStack React Query
- Vite build tool
- CSS3 for styling

## Setup

### Prerequisites

- Node.js 16+ and npm

### Installation

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Configure API endpoint (if needed):

Edit `src/services/api.js` to change the backend URL (default: http://localhost:8001)

### Development

Start the development server:

```bash
npm run dev
```

The application will be available at: http://localhost:5173

### Build for Production

```bash
npm run build
```

Build output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable components
│   │   ├── Navbar.jsx
│   │   ├── Layout.jsx
│   │   ├── ProtectedRoute.jsx
│   │   └── UnauthorizedVehicleTracker.jsx
│   ├── pages/              # Page components
│   │   ├── Login.jsx
│   │   ├── Signup.jsx
│   │   ├── Home.jsx
│   │   └── AuthorizedVehicles.jsx
│   ├── contexts/           # React contexts
│   │   └── AuthContext.jsx
│   ├── services/           # API services
│   │   └── api.js
│   ├── App.jsx            # Main app component
│   ├── main.jsx           # Entry point
│   └── index.css          # Global styles
├── public/
├── index.html
├── package.json
└── vite.config.js
```

## Features Breakdown

### Home Page (Dashboard)

- **Camera Grid**: 2x2 layout displaying 4 camera feeds
  - Camera 1 (Top Left): Entrance camera with LPR
    - Green boxes: Authorized vehicles
    - Red boxes: Unauthorized vehicles
    - Yellow boxes: Vehicles without visible plates
  - Cameras 2, 3, 4 (Monitoring): Vehicle detection only
    - Blue boxes: All detected vehicles
- **Unauthorized Vehicle Tracker**: Real-time table showing
  - Tracking ID
  - License plate (if detected)
  - Vehicle color and model
  - Camera path visualization
  - Timestamps (first/last seen)
  - Active/inactive status
  - Auto-refresh every 5 seconds

### Authorized Vehicles Page (Admin Only)

- View all authorized vehicles in a table
- Add new authorized vehicles
- Edit vehicle information
- Delete vehicles
- Search and filter capabilities

### Authentication

- Login with username or email
- User registration
- JWT token-based authentication
- Persistent sessions (localStorage)
- Automatic token refresh

## API Integration

The frontend communicates with the backend API at `http://localhost:8001/api`

### Main Endpoints Used:

- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user
- `GET /api/cameras` - List available cameras
- `GET /api/stream/{camera_id}` - Video stream
- `GET /api/vehicles` - List authorized vehicles
- `POST /api/vehicles` - Add authorized vehicle
- `PUT /api/vehicles/{id}` - Update vehicle
- `DELETE /api/vehicles/{id}` - Delete vehicle
- `GET /api/tracking/unauthorized` - Get tracking data
- `GET /api/tracking/unauthorized/active` - Get active tracking only

## User Roles

### Regular User
- View camera feeds
- View unauthorized vehicle tracking

### Admin User
- All regular user permissions
- Manage authorized vehicles (CRUD operations)

## Development Notes

### Authentication Flow

1. User logs in → receives JWT token
2. Token stored in localStorage
3. Token added to all API requests via Axios interceptor
4. On 401 response → redirect to login

### Video Streaming

- Uses MJPEG format for simplicity
- Stream URL includes auth token as query parameter
- Auto-reconnects on connection loss

### State Management

- React Context for authentication state
- React Query for server state caching
- Local state for component-specific data

## Troubleshooting

### Video streams not loading
- Check backend is running (http://localhost:8001)
- Verify video files exist in `src/data/raw_videos/`
- Check browser console for CORS errors
- Ensure auth token is valid

### Login issues
- Clear localStorage and try again
- Check backend is running
- Verify credentials are correct

### Build errors
- Delete `node_modules` and `package-lock.json`
- Run `npm install` again
- Check Node.js version (16+ required)
