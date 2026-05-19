WIP/beta lock-in thermography functionality for P1/P3 thermal camera by @hazardousvoltage

The idea and skeleton of the lock-in functionality came from [GitHub - diminDDL/IR-Py-Thermal: Python Thermal Camera Library](https://github.com/diminDDL/IR-Py-Thermal)

This can be used to detect very small temperature changes, such as leakage in electronic components, even below the noise floor of your thermal camera.  It does so by integrating images over a long-ish period of time, while controlling the power flow into the target.

Some external hardware is required, in my case I used an ESP32C3 board.  All it needs to do is read '1' or '0' from a serial port and set a GPIO pin accordingly.  You can then use this output to switch a MOSFET or similar to control the device under test.

I only tested with P3, if anyone can test with P1 and confirm whether it works or not, thanks!

New command-line options and defaults are:
```
`--serial-port` /dev/ttyACM0    Set the serial port used for the controller
`--baud-rate` 115200            Baud rate for serial port
`--period` 1.0                  Period (seconds) of integration cycle
`--integration` 60.0            Time in seconds over which to integrate
`--invert` False                Boolean: invert control output
```
Once the viewer is running and sees your thermal image, press 'l' to run the lock-in.  It will start toggling the output and integrating, showing real-time progress as it goes.  The four quadrants of the output window are, on the left, the in-phase and quadrature integrations, and on the right, the amplitude and phase.  I've tried to make use of sane normalization and color maps to make these look reasonable.

This image is of the ESP32 board which is acting as the lock-in controller.  The added dissipation in the regulator in phase with the output, as well as in the onboard LED and its resistor can be clearly seen.

![esp32-lockin.png](screenshots/esp32-lockin.png)


This is ESP32 again, but with the --invert option reversing the phase of the stimulus.

![esp32-lockin-invert.png](screenshots/esp32-lockin-invert.png)


This one is a 33K resistor connected to the output, being driven with 3.3V from the ESP32.  The power dissipation in this resistor while it's switched on is 330uW, and the resultant heating is clearly visible here.  A second unpowered resistor is next to the target as a control.

![resistor-lockin.png](screenshots/resistor-lockin.png)
