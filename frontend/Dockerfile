#FROM node:18 AS build
#WORKDIR /app
#COPY package.json ./
#COPY public ./public
#COPY src ./src
#RUN npm install
#RUN npm run build

#FROM nginx:alpine
#COPY --from=build /app/build /usr/share/nginx/html
#EXPOSE 80
#CMD ["nginx", "-g", "daemon off;"]

# Build React app
FROM node:18 AS build
WORKDIR /app

# Install deps
COPY package*.json ./
RUN npm install

# Copy app source
COPY . .
RUN npm run build

# Serve with Nginx
FROM nginx:alpine
# Copy built React files from Stage 1
COPY --from=build /app/build /usr/share/nginx/html

# Custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

