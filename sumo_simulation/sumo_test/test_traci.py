import os, sys
import random
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=12, kernel_initializer='glorot_uniform', activation='relu'))
#model.add(Dense(50, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='glorot_uniform', activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse',])

model.save("testNN")

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

def removeID(lista,x):
	if (x!=0) and (x in lista):
		lista.remove(x)
	return lista

def defineScenario(dbase,vehicleID):
	dbase[vehicleID]={'nPassenger': random.randint(0,5)}

t=traci.connect(27910)
#t.simulation.subscribe()
lista_attivi=[]

while(True):
	l=t.simulationStep()
	#t.simulation.getCollisions()
	#n=t.simulation.getCollidingVehiclesIDList()
	#print(n)
	#n=t.simulation.getCollidingVehiclesNumber()
	#n1=t.simulation.getColliderIDs()
	collisions = t.simulation.getCollisions()
	positions = t.person.getAllSubscriptionResults()
	
	
	'''print(t.simulation.getLoadedIDList())
	lista_attivi=lista_attivi +[x for x in t.simulation.getLoadedIDList()]
	[print(x) for x in t.simulation.getArrivedIDList()]
	print(lista_attivi)
	for x in t.simulation.getArrivedIDList():
		lista_attivi=removeID(lista_attivi,x)
	for i in range(len(lista_attivi)):
		v1=lista_attivi[i]
		for j in range(i,len(lista_attivi)):
			v2=lista_attivi[j]
			print(f"V1 {v1} POSV1 {t.person.getPosition(v1)} V2 {v2} POSV2 {t.vehicle.getPosition(v2)}")'''
	
	for v1 in t.vehicle.getIDList():
		for p1 in t.person.getIDList():
			posV=t.vehicle.getPosition(v1)
			posP=t.person.getPosition(p1)
			distance = t.simulation.getDistance2D(posV[0],posV[1],posP[0], posP[1], isDriving=True)
			minGap = t.vehicle.getMinGap(v1)
			if minGap * 1.5 >= distance: 
				print(f"DISTANCE {v1} {p1}: {distance} {minGap * 1.5 >= distance} {t.vehicle.getPersonNumber(v1)}")
				break

	#print(f"ARRIVED: {list(t.simulation.getArrivedIDList())}")
	#for v in t.simulation.getLoadedIDList():
		#print(v)
		#print(f"NEIGHBOURS {v}: {t.vehicle.getNeighbors(v,1+2+0)}")
	#print(positions)
	#if "3" in t.simulation.getLoadedIDList(): print(t.vehicle.getSignals("3"))
	
	if collisions!=(): 
		print(collisions[0])
		collider = collisions[0].__attr_repr__('collider').split("=")
		victim = collisions[0].__attr_repr__('victim').split("=")
		print(f"Collider: {collider} Victim: {victim} ")
		posV=t.vehicle.getPosition(str(collider[1]))
		posP=t.person.getPosition(str(victim[1]))
		print(f"ID: {collider[1]} POSITION COLLIDER CAR: {posV} POSITION VICTIM PEDON: {posP}")
		print(f"DISTANCE: {t.simulation.getDistance2D(posV[0],posV[1],posP[0], posP[1])}")
		print(f"NEIGHBOURS {collider[1]}: {t.vehicle.getNeighbors(collider[1],3)}")
		print(f"MINGAP {collider[1]}: {t.vehicle.getMinGap(collider[1])}")
		print(f"PRESENT VEHICLE: {list(t.vehicle.getIDList())} {t.vehicle.getIDCount()}")
		print(f"PRESENT PED: {list(t.person.getIDList())} {t.person.getIDCount()}")
		print()
		#print(t.vehicle.getSignals("3"))
		#print(f"POSITION 3: {t.vehicle.getPosition('3')}  POSITION 5: {t.person.getPosition('5')}")
		#break
	#if len(n1)>0: print(n1)
t.close()

#traci.start([sumolib.checkBinary('sumo'),
#    "-n", "test.net.xml",
#    "-r", "cars2_mod.rou.xml,pedestrians2.rou.xml",
#    "--collision.check-junctions",
#    "--no-step-log",
#	"--pedestrian.striping.dawdling", "5"
#    ])

#sumo-gui --step-length 0.05 -n test.net.xml -r cars2_mod.rou.xml,pedestrians2.rou.xml --pedestrian.striping.dawdling 5 --remote-port 27910 --start -Q 	--collision.mingap-factor 50 --collision.check-junctions --collision.action warn