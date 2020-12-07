# EE-629-Project
Machine lifetime prediction using Pi Camera and Google cloud.

Code Files:
* upload_live_param1.py = Machine Learning and Cloud uploading.
* Image Processing/arasm_live.py = Capturing image, Processing and creating Dataset based on Image processing output.


Technologies Used:

* Internet of Things.
  - Camera module with help of Image Processing acts as Sensors.
  - Machine Learning module Processes the data.
  - Results obtained from Machine Learning are then stored in Google Cloud using Googel API.
  - Augmented Reality based application allows users to access the Data on their personal Android or IOS devices.

Capture Images every 5 seconds using Camera module.

* Image processing.

Processing Multiple Analog Output devices such as Gauge, Thermometer, Ammeter/Voltmeter digits, etc.
Libraries Used: 
  - OpenCV 
  - Tesseract
  - PIL.

<p><img src='Outputs/ocr.jpg' />
  
<p><img src='Image_Processing/images/gauge-1.jpg' />
 
<p><img src='Image_Processing/images/screen-prompt.jpg' />
  
<p><img src='dst1.jpg' />

* CSV File generation

Generate CSV File using Pandas library.

<p><img src='Outputs/csv.jpg' />

* Regression Prediction.
Calculate Prognosis and Diagnosis Based on collected data using ARIMA.
<p><img src='Outputs/prognosis.png' />
<p><img src='Outputs/diagnosis.png' />

* Time-Series Analysis.
Generate Graph based on Predictions.
<p><img src='Outputs/op.jpg' />
<p><img src='Graph/graph2.png' />

* Google API Configuration.
Genreate Oauth2.0 Client ID for Web Application.

https://console.developers.google.com/apis/dashboard 

- Steps:
  - Create a Project
  - Click on 'Enable API's and Services'
  - Search for Google Drive API and enable API
  - Click on 'Credentials'
  - Create 'OAuth 2.0 Client IDs'
  - Download JSON file after configuration
  
<p><img src='Outputs/gapi.jpg' />
 
(Download JSON file and create Pickle file containing Token which bypasses authentication for frequent use.)
<p><img src='Outputs/pickle.jpg' />

* Augmented Reality.
Create QR code for Collecting Data from cloud.

https://developer.vuforia.com/vui/develop/licenses

- Steps:
  - Click on 'Target Manager'.
  - Create Dataset and Click on 'Add Target'.
  - Save information and Download image.

<p><img src='Outputs/Machine1_cloud.jpg' />

Unity, Vuforia Application

<p><img src='Outputs/unity.jpg' />

Install app in your android/IOS device.

Hover mobile camera over QR code and you can supervise and watch real-time graph generation and information regarding Machine state and Lifetime Prediction.
<p><img src='Outputs/1.jpeg' />
<p><img src='Outputs/2.jpeg' />
