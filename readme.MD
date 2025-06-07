conda create --name videosummary38 python=3.8.0
conda activate videosummary38
python app.py

Install the required libraries with the specific version mentioned in req.txt

=====================================================================================================================

The TypeError: the JSON object must be str, bytes or bytearray, not NoneType error usually occurs when the get_transcript() function is unable to retrieve the transcript for certain YouTube videos, which could be due to a few common reasons:

No Transcript Available: Some YouTube videos, like music videos, private videos, or certain restricted videos, don’t have transcripts, and get_transcript() returns None for these videos.

Video Permissions: Videos marked as private, age-restricted, or country-restricted might not allow transcript retrieval.

API Limitations: If you're using an API or library for get_transcript(), it may have limitations or require certain settings to access transcripts.

=====================================================================================================================

Google Translate has limits on the amount of text you can send for translation. If your summary becomes too long, the Google Translate may fail to process it, which explains why it works for shorter videos but fails for longer ones.






 
