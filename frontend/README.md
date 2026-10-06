# TonerHound Frontend

Web interface for TonerHound deterministic document evidence grounding. Built with Next.js 16 (App Router), TypeScript, Tailwind CSS, and `react-pdf`.

## Getting Started

1. Ensure the TonerHound FastAPI backend is running:
   ```bash
   cd ..
   uv run uvicorn backend.server:app --reload --port 8000
   ```

2. Run the frontend development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Configuration

Set `NEXT_PUBLIC_API_URL` in `.env.local` or deployment environment variables to point to your backend API URL (defaults to `http://localhost:8000`).
