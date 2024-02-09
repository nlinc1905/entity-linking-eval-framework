mlflow server


Notes:
model is not making automated decisions, it's providing a ranked list
but it's being evaluated on how well it predicts links (how well it would serve as an automated record linker)
the model's true output is a ranked list

should not evaluate model on how well it predicts links
should evaluate it on how well it sets up the human to do their task

need a ranker model, right now it's a simple sort
model feedback comes from humans labeling output as 1 or 0, but should come from the rank instead
we know there are inputs errors, so there could be labeling errors too, and these should not 
be able to impact the classifier.  The ranker would serve as a buffer: provide soft labels to the 
classifier once a user consensus has been reached.  Until then, re-sort the results and evaluate 
the rankings.
