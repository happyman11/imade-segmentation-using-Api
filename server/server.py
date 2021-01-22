from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import random

# Initialize the Flask application
app = Flask(__name__)
# segment function
def segment(image):

    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10,    cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    masked_image = np.copy(image)
    masked_image = masked_image.reshape((-1, 3))
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 0]
    masked_image = masked_image.reshape(image.shape)


    

    return (segmented_image,masked_image)
    
    

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    image,masked_image= segment(img)
    # build a   response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    image_name = 'image_{}.jpg'.format(random.randint(1,1000))
    image_name1 = 'image_{}.jpg'.format(random.randint(1,1000))
    cv2.imwrite(image_name, image)
    cv2.imwrite(image_name1, masked_image)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == '__main__':
    app.run(port = 5000, debug=True)

