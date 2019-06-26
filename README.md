# plantclef2019_challenge
Details and supplement material for plantclef2019 challenge

## Training and testing lists
The related list can be found in 'training_and_validation_list'

- clef2019_classid_map_to_index.txt  
The mapping of classid (species) provided by PlantClef2019 to index ranging from 0-9999  

- clef2019_family_map_to_index.txt  
The mapping of family label provided by PlantClef2019 to index ranging from 0-248  

- clef2019_genus_map_to_index.txt  
The mapping of genus label provided by PlantClef2019 to index ranging from 0-1780  

- clef2019_multilabel_test_256046.txt  
The annotation of test data in the format of: *img_path family_label genus_label species_label*  

- clef2019_multilabel_train_256046.txt  
The annotation of training data in the format of: *img_path family_label genus_label species_label* 

- clef2019_non_plant_list.txt  
Non plant list classify using inceptionV4 trained with PlantClef2016 and Imagenet2012  

- clef2019_test_256046.txt  
The annotation of test data in the format of: *img_path species_label*  

- clef2019_train_250646.txt  
The annotation of training data in the format of: *img_path species_label*  
