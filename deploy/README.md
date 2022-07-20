## Build the image

Check that you are within the `deploy` directory and use the `docker build` command.
```bash
docker build -t captcha:latest .
```

## Run the container

```bash
docker run --rm -p 8000:80 captcha:latest
```