# Combined Dockerfile for both client and server
# Builds the React client and serves it via the Node.js server

# Build stage for client
FROM node:20-alpine AS client-builder

WORKDIR /app/client

# Copy client package files
COPY client/package.json client/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy client source files
COPY client/ ./

# Build the client
RUN npm run build

# Production stage
FROM node:20-alpine

WORKDIR /app

# Copy server package files
COPY server/package.json server/package-lock.json* ./

# Install server dependencies
RUN npm install --production

# Copy server source files
COPY server/ ./

# Copy built client files
COPY --from=client-builder /app/client/dist ./client/dist

# Create data directory
RUN mkdir -p data tmp/uploads

EXPOSE 3002

ENV NODE_ENV=production

CMD ["node", "index.js"]
