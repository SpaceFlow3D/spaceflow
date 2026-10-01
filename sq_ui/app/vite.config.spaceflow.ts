import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { npzOpenPlugin } from './vite.npzOpenPlugin.ts'

// In dev, the browser may load the UI via the machine's hostname. Requests to localhost
// would then hit the user's laptop, not this host. The proxy keeps SpaceFlow same-origin.
const spaceflowTarget = process.env.VITE_DEV_PROXY_SPACEFLOW ?? 'http://127.0.0.1:11438'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), npzOpenPlugin()],
  server: {
    watch: {
      usePolling: true,
      interval: 1000,
    },
    proxy: {
      '/spaceflow': { target: spaceflowTarget, changeOrigin: true },
    },
  },
})
