# Pedestrian-Crossing-and-Traffic-Light-Detection


### Preprocessing
  Positioning of traffic ligths in the AIGO's field of view and zoom in on input images to better recognize distant traffic lights 
  #### Convert the bounding box style to draw lines
  * YOLO Style
    ( center of x, center of y, width, height )
  * labels example
    ```
    0 0.291543 0.288690 0.048363 0.032407
    ```
  * Convert YOLO Style to (x1, y1, x2, y2) Bounding Box Style
      
      ( top left x, top left y, bottom right x, bottom right y )

  * Convert formula
    * [convert and draw lines](https://github.com/icns-distributed-cloud/Pedestrian-Crossing-and-Traffic-Light-Recognition/blob/master/convert_and_draw_bounding_boxes.py)


   #### Result
  <img src="https://user-images.githubusercontent.com/68395698/124228912-d35a3f00-db47-11eb-8b3c-15c1715c9af6.jpg" width=500>


 
