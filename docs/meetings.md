# Meeting log

Copied from TeamFeedback. One entry per meeting.

## DATE (week 2)/ 10 November
Present: Everyone


Decisions:
- Requirements: test accuracy >= 90%, macro F1 >= 0.85, warning below 50% confidence,
  under 2 s per prediction on a laptop CPU, app states it is for single-leaf lab-style photos.
- Deployed model: the 99% CNN (ClearML train task 1bee769f). Field fine-tuning raised field
  accuracy to 22.9% but cut lab accuracy to 85.7%, so it is reported, not deployed.
- Final-setup candidate: strong augmentation trained longer (better on field photos and under blur).

Actions:
- Precious: ResNet18 / residual CNN comparison (#28) by Mon 16 Nov, then report section 3.1
- Amrit: sweep and final setup (#29, #30) by Tue 17 Nov, then report section 3.2
- Maryam and Media: Grad-CAM, app warning, CPU timing (#32), then sections 4 and 5
- Nischal: assign issues, Week 2 milestone, Gantt chart, section 7
- Sara: section 2 final (done), meeting log
