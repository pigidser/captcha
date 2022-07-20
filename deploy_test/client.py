import requests


def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.models.Response: Response from the server.
    """
    
    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response.text


# base_url = 'http://192.168.1.108:8000'
base_url = 'http://localhost:8000'
endpoint = '/predict'
url = base_url + endpoint


with open("../input/captcha0.png", "rb") as image_file:
    prediction = response_from_server(url, image_file)

print(prediction)