# AHP-covid19-ctscans
Positive COVID-19 CT scans from the American Hospital of Paris

The `Anonymized` and `Thorax` folders contain DICOM images (`.dcm` file extension). This is a complex, highly dimensional file format.

There are many softwares for Windows and MAC which open `.dcm` extension files.

The files in all other folders have no file extension at this time. Probably they should have `dcm` extension.

Beware, the whole files are over 1.3 GB.

One of the biggest challenges developers face while building web based medical imaging applications is fast image loading. The main issue here is that medical images are typically stored on the server using an image compression format such as JPEG-2000 or JPEG-LS that web browsers don't know how to decode. 

Use this link <https://chafey.github.io/openjpegjs/test/browser/index.html> to decode JPEG2000 with with WebAssembly. 

Credit goes to Chris Hafey for his brilliant work.  
