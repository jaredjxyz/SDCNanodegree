# Reflection

## Describe the effect each of the P, I, D components had in your implementation.

The higher the P is, the sharper the car will turn when it's far away from the center line.

The higher the I is, the more a car will compensate for a bad steering configuration. This is not very important in this project because the car runs pretty straight.

The higher the D is, the more the car will turn to go forward when it's not already going forward. This is good for dealing with turns, as the car will adjust to the angle of the turn.

Any of these three being two high will make the car swerve around the track.

## Describe how the final hyperparameters were chosen.

I started with P = .1, I = .1, and D = .1. The I ended up being way too big and started making the car swerve really sharply to the right or left immediately, so I dialed back I to .0001.

I tried P = .1, I = .0001, and D = .1, and that worked better but the car was swerving all over the track and sharply, so I halved the P to .05.

I tried P = .05, I = .0001, and D = .1, and that worked even better but the car wasn't turning sharp enough on corners.

I decided I needed to turn P back up to .1, but then D needed to be higher to compensate, so I set D to .3, which worked better, and then .5, which worked really well, so I stayed with that.
