Flask API entry point to microservice image fraud detection/ classifier. consumes URLs and outputs JSON analytics


**Feature list**
| Feature | Description | Status |
| --- | ----------- | ----------- |
| API Endpoint | Integration of main image analysis module | In Development |
| Copy-move Detection | Using SIFT and DBSCAN for passive image forgery detection | In Testing |
| Splice Detection  | Utilises ML methodologies | In Research |
| Face Recognition | Flagging images that contain face(s) of persons | In Research |
| GAN Detection | used to detect face manipulation, deep fakes or ai rendered faces | TBS |
| API Security | If no login then rate limits on api resource (threading) or Oath with email/gmail and more | In Testing |
| Chrome Plugin | Javascript front end to scrape images and send urls to API for report and visual detection | In Testing |
| Benford Law Analysis (experimental) | Applies the statistical rules of Benfords Law to image vector analysis | In Research |
| Sys admin API interface | API endpoints for database reading and  configuration settings | TBS |
| GUnicorn | Implement GUnicorn for production  concurrency | In Research |
| NGinx | Works in tandem with GUnicorn. Serves requests to GUnicorn and static to itself | In Research |
| Web Page | Create front end web page to allow user to upload or paste link and get report in response (makes call to microservices API). Optional analysis type available with API gateway | TBS |
| S3 Offloading | Offload weights to AWS S3 and load on app start up | TBS |
| API Gateway | Implement API gateway after development of main microservices (route mapping/JSON consolidating) | TBS |










