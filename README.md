# Augmented-Variational-Autoencoders-for-Collaborative-Filtering-with-Auxiliary-Information
Augmented Variational Autoencoders for Collaborative Filtering
with Auxiliary Information (Lee et. al.) (CIKM-17) </br>
<a href=https://dl.acm.org/doi/10.1145/3132847.3132972> Paper Link 1 </a>
<a href=https://aailab.kaist.ac.kr/xe2/board_international_conferences/17819> Paper Link 2 </a>

We attached the source code of the model and data reader (VAE-CF and DataReader) </br>
Because the source codes were implemented in an earlier version of TF, it will take time to convert to a newer version. </br>
Please check the implementation of internal functions (prepare_model, build_encoder, build_decoder, build_objective_fn). </br>

example of data format </br>
user_id::item_id::feedback 
