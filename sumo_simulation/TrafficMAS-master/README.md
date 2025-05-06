[![Build Status](https://travis-ci.org/baumfalk/TrafficMAS.svg?branch=master)](https://travis-ci.org/baumfalk/TrafficMAS)
[![Coverage Status](https://coveralls.io/repos/baumfalk/TrafficMAS/badge.png)](https://coveralls.io/r/baumfalk/TrafficMAS)

# TrafficMAS
TrafficMAS, the code

##How to install##
###Install SUMO###
In general: see the [SUMO wiki](http://sumo.dlr.de/wiki/Installing) about installing SUMO
+ Windows  
    Get SUMO binaries from [DLR.de](http://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/).
+ Linux  
    You can do 
    >	`apt-get install sumo`
    but note that these versions are often very out of date. A good alternative would be to compile from source.
+ OSX  
	SUMO needs to be compiled from source on OS X. We succeeded in doing it by following these instructions:
	+	1. Install [Homebrew](http://brew.sh/)
	+	2. Install [XQuartz](http://xquartz.macosforge.org/landing/), you need this for the GUI at least, regular SUMO might work without. 
		+ At this point you can try to run this script for automatic install: http://pastebin.com/G206kgbh
	+	3. Download [SUMO source](http://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/)
	+	4. Use Homebrew to install the following dependencies: fox (only for GUI), gdal, proj, xerces-c.
			This can be done by running `brew install xerces-c gdal proj fox`
	+	5. Set the following flags through the environment variables:  
			`export CXXFLAGS="-I/opt/X11/include"`  
			`export LDFLAGS="-framework OpenGL -framework GLUT -L/usr/X11/lib -L/usr/X11R6/lib -lpython2.7"`
	+	6. Now run `./configure --with-python` in the folder where you downloaded the source.
	+	7. Finally run `make` to compile and `make install` to set paths to the binairies.


You can test if sumo was installed succesfully by opening a terminal and typing
>    `sumo`

It should then return something like

		SUMO Version 0.22.0
		Copyright (C) 2001-2014 DLR and contributors; http://sumo.dlr.de
		License GPLv3+: GNU GPL Version 3 or later <http://gnu.org/licenses/gpl.html>
		Use --help to get the list of options.`

If it doesn't work, you should add the sumo executable to your path variable. Alternatively,
you can give the link to the sumo executable as a parameter to the TrafficMAS program.

###Compiling The Code###
The easiest way is to use Eclipse:

1. Clone the code  
2. Create a new workspace in the cloned folder  
3. New > Add Project. Use TrafficMAS as name, so Eclipse can automatically set things up for you. 
4. Go to 'window' and at 'show view' select ant. 
5. An ant window should have popped up on the right, click on "Add buildfile" (the ant icon with the green '+' symbol) 
6. Run the buildfile by clicking the run button in the ant window.
7. Right click (RMB) the project, select 'properties' and go to 'builders'. 
8. Select import and browse to the build.xml file located in /TrafficMAS
9. Now the latest TraaS.jar has been build into the /lib directory!
10. Right click the TrafficMAS project, "Properties > Java Build Path, go to the Libraries tab.
11. Select the TraaS.jar file and press edit. Now select the recently build TraaS.jar in /lib.
12. Done!  

If you also want to run our tests, do the following
1. Add JUnit4 to the libraries. The easiest way to do this is by RMB > New JUnit test, and have Eclipse
add the JUnit library itself.
2. (Optional) Add [Infinitest](http://infinitest.github.io/) to Eclipse. This allows for automatic test running when you changed the source

###Running The Code###
If you are using Eclipse, you can easily run the code.
  
1. Press the Run button. You'll get the message `At least four arguments needed`  
2. Now click the small triangle next to the Run button.  
3. In the arguments tab, add the parameters. The first parameter is the main directory, the second is the main TrafficMAS, the third one is the location for sumo(-gui) and the fourth parameter is the seed. For example `./sim/ hello.mas.xml sumo hello.sumocfg 1337` states that the main directory is `./sim`, that the TrafficMAS config file is `hello.mas.xml`, the path for sumo is `sumo` (because it was added to the path variable. If not, we would have written something like `/usr/bin/sumo` or `C:\sumo\sumo.exe`), that the sumo config file is `hello.sumocfg` and that the seed the program will use is `1337`. If you want to also see a graphical interface for sumo, replace `sumo` with `sumo-gui`.  
4. If you now press the Run button again, you should see output in the console screen (and a sumo gui if you used that as a parameter).

###Changing the program behavior###
In addition to the standard SUMO stuff (roads, edges etc), you can also change the agent profile distributions. Their xml location is specified in MAS.xml. In this xml, `spawn-probability` denotes the probability that an agent spawns per tick. For every agent role (or profile), the `dist` parameter denotes the relative spawn probability for this role. That is, if `dist=0.3` for the Normal role, then *if* an agent spawns, it will be of the Normal profile 30% of the time. 
