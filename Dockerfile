# Pull the minimal Ubuntu image
FROM ubuntu

# Install Nginx
RUN apt-get -y update && apt-get -y install nginx

# Copy the Nginx config
COPY nginx.conf /etc/nginx/sites-available/default
COPY index.html /usr/share/nginx/html
COPY train.png /usr/share/nginx/html
COPY test.png /usr/share/nginx/html
COPY predict.png /usr/share/nginx/html
# Expose the port for access
EXPOSE 80/tcp

# Run the Nginx server
CMD ["/usr/sbin/nginx", "-g", "daemon off;"]
