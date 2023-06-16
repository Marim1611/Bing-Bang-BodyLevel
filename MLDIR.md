## 📜 The MLDir Manifesto
□ Each stage in the ML pipeline should be a separate directory.

□ For any pipeline stage, each alternative that implements that stage should be in a separate directory within that stage's folder. For example, different models or features in the model or feature extraction stages respectively.

□ Any implementation of a stage is a set of functions and it is defined in a .py file within the directory of the specific alternative of the stage. If a function is used over different alternatives then the .py file can be at the same level as the directories of the alternatives.

□ Notebooks are only for demonstration or running entire pipelines (e.g., training and hyperparameter tuning). Functions are never defined in them.

□ Any notebook cell should be preced with a clear heading that describes what it does.


□ Call the training, validation and testing data x_train, x_val and x_test respectively. x_val is x_val even if x_test doesn't exist (yet). The variable name may be appended with an _{letter} to indicate the stage of the pipeline that has produced it.

□ Include a Saved directory for saving trained models, figures and other artifacts.

□ If a pipeline stage may benefit from visualizion then provide a method for that.

□ Similar or related visualizations should be grouped together in the same figure whenever possible.

□ Logging should be implemented for every pipeline.

□ Once the experimentation phase is over (converged on a pipeline with fixed hyperparameters), the pipeline should be implemented in a single .py file with a single pipeline function. The project, except for this file can be archived at this point after producing a requirements.txt


