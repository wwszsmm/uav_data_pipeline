# Road Segmentation Model Evaluation

## Model Performance

The road segmentation model achieved approximately:

- Mask mAP@0.5: 0.902
- Mask mAP@0.5:0.95: 0.778

The confusion matrix showed limited confusion between classes. Most remaining errors were missed detections rather than incorrect class predictions.

## Manual Sample Analysis

Six representative validation samples were reviewed:

- 2 model error cases
- 2 annotation quality cases
- 2 good prediction cases

### Model Errors

The main model errors occurred in distant and small target regions:

- A distant no-parking zone was missed.
- A distant road region was missed.

This indicates that small objects and targets far from the UAV camera remain the main weakness of the model.

### Annotation Quality Issues

Some images contained incomplete reference annotations because different object classes had been annotated in separate tasks.

In the reviewed cases, the model output was consistent with the existing annotations, but the annotations did not completely describe all visible target classes.

Because of this issue, the reported validation metrics may not fully represent the model's actual performance.

### Good Examples

The model performed well on clear and larger road regions:

- Road boundaries were segmented accurately.
- Multiple road regions were correctly detected in complex scenes.

## Conclusion

The model provides reliable segmentation for clear and medium-to-large road regions. Its main limitation is missed detection of distant or small targets.

Future improvement should prioritize annotation completeness and additional training samples containing distant targets, rather than only adjusting model parameters.