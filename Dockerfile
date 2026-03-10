FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy site files
COPY index.html /usr/share/nginx/html/index.html
COPY public/    /usr/share/nginx/html/public/

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
