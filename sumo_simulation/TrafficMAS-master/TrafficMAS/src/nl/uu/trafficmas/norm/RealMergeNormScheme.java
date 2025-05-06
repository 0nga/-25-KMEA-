package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import nl.uu.trafficmas.exception.InvalidDistanceParameter;
import nl.uu.trafficmas.exception.InvalidParameterCombination;
import nl.uu.trafficmas.exception.InvalidVPrimeParameter;
import nl.uu.trafficmas.exception.InvalidVelocityParameter;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class RealMergeNormScheme extends NormScheme {

	private Sensor mainSensor;
	private Sensor rampSensor;
	private Sensor laneSensor;
	public Sensor mergeSensor;
	private double LastCarMergePointTime 	= -1;
	private Set<String> tickedAgents;
	private static List<Sensor> sensors;
	public static final double MAX_VELOCITY = (80/3.6);
	public static final double TIME_BETWEEN_CARS = 4;
	//TODO: Remove or implement this.
	//public static final double DISTANCE_BETWEEN_CARS = 9;
	private static final double PRECISION = 100;
	public static final double MERGEPERCENTAGE = 0.9;
	
	/**
	 * first sensor: main road
	 * second sensor: ramp
	 * third sensor: after merge
	 * @param sensorList the list of the sensors, used to determine how to merge.
	 */
	public RealMergeNormScheme(String id, SanctionType sanctionType, List<Sensor> sensorList) {
		super(id,sanctionType,sensorList);
		sensors		= sensorList;
		mainSensor 	= sensorList.get(0);
		rampSensor 	= sensorList.get(1);
		mergeSensor	= sensorList.get(2);
		laneSensor	= sensorList.get(3);
		tickedAgents= new HashSet<String>();
	}
	
	@Override
	public List<NormInstantiation> instantiateNorms(RoadNetwork rn, int currentTime, Map<String, AgentData> currentOrgKnowledge) {
		double vmax = MAX_VELOCITY;
		
		List<AgentData> mainList = mainSensor.readSensor();
		List<AgentData> rampList = rampSensor.readSensor();
		
		Collections.sort(mainList, new MergeNormAgentDataComparator());
		Collections.sort(rampList, new MergeNormAgentDataComparator());

		// only for new agents
		removeTickedAgents(mainList);
		removeTickedAgents(rampList);
		
		// calculate merged list
		List<AgentData> outputList = mergeTrafficStreams(mainList, rampList, mainSensor.getEndPosition(), rampSensor.getEndPosition());
		// tick the new agents: we processed them
		for (int i = 0; i < outputList.size(); i++) {
			tickedAgents.add(outputList.get(i).id);
		}
		
		// calculate norm and arrival time for first car
		List<NormInstantiation> normInstList = new ArrayList<NormInstantiation>();
		
		double lastCarArrivalTimeMergePoint = LastCarMergePointTime;
		
		
		// calculate norm and arrival times for other cars
		List<Double> carArrivalTimes = new ArrayList<Double>();
		for (int i = 0; i < outputList.size(); i++) {
			
			AgentData currentCar 	= outputList.get(i);
			lastCarArrivalTimeMergePoint = carNormInstantiation(rn, vmax,
					TIME_BETWEEN_CARS, normInstList,
					lastCarArrivalTimeMergePoint, currentCar, currentTime);
			carArrivalTimes.add(lastCarArrivalTimeMergePoint);
		}
		LastCarMergePointTime 	= lastCarArrivalTimeMergePoint;
		return normInstList;
	}

	private double carNormInstantiation(RoadNetwork rn, double vmax,
			double timeBetweenCars, List<NormInstantiation> normInstList,
			double prevCarArrivalTimeMergePoint, AgentData currentCar, int currentTime) {
		RealMergeNormInstantiation ni;
		double lastSpeed;
		double lastCarMergePoint;
		double distRemaining;
		double acceleration;
		double newPrevCarArrivalTimeMergePoint = prevCarArrivalTimeMergePoint;
		
		lastCarMergePoint 		= rn.getEdge(currentCar.roadID).getRoad().length;
		distRemaining			= lastCarMergePoint - currentCar.position;
		if(prevCarArrivalTimeMergePoint == -1) {
			lastSpeed	 = vmax;
			acceleration = (lastSpeed < currentCar.velocity) ? currentCar.deceleration : currentCar.acceleration;
		} else {
			
			double travelTime = prevCarArrivalTimeMergePoint-currentTime;
			double posAtArrivalTime = currentCar.velocity* (timeBetweenCars + travelTime)+ currentCar.position;
			acceleration 			= (posAtArrivalTime < lastCarMergePoint) ? currentCar.acceleration : currentCar.deceleration;
			lastSpeed 				= findBestSpeedTime(currentCar.velocity, acceleration, distRemaining, travelTime, timeBetweenCars );
			//distanceSpeed			= findBestSpeedDistance(currentCar.velocity, acceleration, distRemaining, currentTime, newPrevCarArrivalTimeMergePoint, DISTANCE_BETWEEN_CARS);
			if(Double.isNaN(lastSpeed)) {
				lastSpeed = vmax;
			}
		}
		try {
			newPrevCarArrivalTimeMergePoint = currentTime + getTravelTime(currentCar.velocity, acceleration, distRemaining, lastSpeed);
		} catch (InvalidVPrimeParameter e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidVelocityParameter e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidDistanceParameter e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidParameterCombination e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		ni = new RealMergeNormInstantiation(this, currentCar.id);
		double correctedLastSpeed = Math.round(lastSpeed*PRECISION)/PRECISION;
		//double distanceCorrectedSpeed = Math.round(distanceSpeed*PRECISION)/PRECISION;
		// TODO: Dynamicize
		// If TIME_BETWEEN_CARS distance with this speed means that the gap between cars is less than 2 meter,
		// Calculate new speed, which insures the 2 m minGap.
		// Argh, also keep in mind that cars have a certain size, currently 7 m. So 9 m gap between the two positions.
		//if(TIME_BETWEEN_CARS*correctedLastSpeed < DISTANCE_BETWEEN_CARS){
			//ni.setSpeedAndLane(distanceCorrectedSpeed, 0);
		//} else {
		ni.setSpeedAndLane(correctedLastSpeed, 0);
		//}
		normInstList.add(ni);
		return newPrevCarArrivalTimeMergePoint;
	}

	public static double findBestSpeedTime(double velocity, double acceleration,
			double distRemaining, double lastCarArrivalTimeMergePoint,
			double timeBetweenCars) {
		
		double vprime;
		
		double time = lastCarArrivalTimeMergePoint+timeBetweenCars;
		
		double firstSqrt 		= (acceleration*acceleration*time*time);
		double secndSqrt		= (2*acceleration*distRemaining);
		double thridSqrt		= (2*acceleration*time*velocity);
		double afterSqrt		= (acceleration*time+velocity);
		
		double sqrt = firstSqrt-secndSqrt+thridSqrt;
		
		double posResult = Math.sqrt(sqrt)+afterSqrt;
		double negResult = -Math.sqrt(sqrt)+afterSqrt;
		
		if (velocity * time > distRemaining){
			vprime = posResult;
		} else {
			vprime = negResult;
		}
		
		return vprime;
	}
	
	public static double findBestSpeedDistance(double velocity, double acceleration,
			double distRemaining, double currentTime, double lastCarArrivalTimeMergePoint,
			double distanceBetweenCars) {
		
		double v 	= velocity;
		double a 	= acceleration;
		double m 	= distRemaining;
		double t1	= currentTime;
		double t2	= lastCarArrivalTimeMergePoint;
		double t 	= t2 - t1;
		double d	= distanceBetweenCars;
		double vprime;
		
		double firstSqrt 		= (a*a*t*t);
		double secndSqrt		= (2*a*d);
		double thirdSqrt		= (2*acceleration*m);
		double forthSqrt		= (2*a*t*v);
		double afterSqrt		= (a*t+v);

		double sqrt = firstSqrt+secndSqrt-thirdSqrt+forthSqrt;
		
		vprime = Math.sqrt(sqrt)+afterSqrt;
		
		return vprime;
	}

	public static double getTravelTime(double velocity, double acceleration,
			double distRemaining, double vprime) throws InvalidVPrimeParameter, InvalidVelocityParameter, InvalidDistanceParameter, InvalidParameterCombination {
		
		double accelerationTime = Math.abs((vprime - velocity)/acceleration);
		double accelerationDist = accelerationTime*(vprime+velocity)/2;
		
		boolean negativeVelocity				= (velocity < 0);
		boolean negativeVPrime					= (vprime < 0);
		boolean positiveSpeedDeltaNegativeAccel = velocity < vprime && acceleration <= 0;
		boolean negativeSpeedDeltaPositiveAccel = velocity > vprime && acceleration >= 0;
		boolean notEnoughDistRemaining = (accelerationDist > distRemaining);
		if(notEnoughDistRemaining)
			throw new InvalidDistanceParameter();
		if(negativeVelocity)
			throw new InvalidVelocityParameter();
		if(negativeVPrime)
			throw new InvalidVPrimeParameter();
		if(positiveSpeedDeltaNegativeAccel || negativeSpeedDeltaPositiveAccel)
			throw new InvalidParameterCombination();
		
		double remainingTime = (distRemaining-accelerationDist)/vprime;
		return accelerationTime+ remainingTime;
	}

	private void removeTickedAgents(List<AgentData> mainList) {
		Iterator<AgentData> iter = mainList.iterator();
		while (iter.hasNext()) {
		    AgentData ad = iter.next();
		    if (tickedAgents.contains(ad.id))
		        iter.remove();
		}
	}

	public static List<AgentData> mergeTrafficStreams(List<AgentData> mainList, List <AgentData> rampList, double mainRoadEnd, double rampRoadEnd) {
		List<AgentData> tempMainList = new ArrayList<AgentData>(mainList);
		List<AgentData> tempRampList = new ArrayList<AgentData>(rampList);
		List<AgentData> outputList = new ArrayList<AgentData>();
		boolean cond = true;
		AgentData m = null;
		AgentData r = null;
		double t_r = Double.POSITIVE_INFINITY;
		double t_m = Double.POSITIVE_INFINITY;
		while(!(tempRampList.isEmpty() && r== null)) {
			if(r==null) {
				r = tempRampList.remove(0); 
			}
			
			if(tempMainList.isEmpty() && m== null) {
				outputList.add(r);
				r = null;
			} else {
				do {
					
					if(cond) {
						m = tempMainList.remove(0);
						double distanceUntilEndOfMainRoad = mainRoadEnd - m.position;
						t_m = distanceUntilEndOfMainRoad/m.velocity;
						cond = false;
					}
					double distanceUntilEndOfRamp = rampRoadEnd - r.position;
					t_r = distanceUntilEndOfRamp / r.velocity;
					if(t_m > t_r) {
						outputList.add(r);
						r = null;
					} else {
						outputList.add(m);
						cond = true;
						m = null;
					}
				}while(!tempMainList.isEmpty() && cond);
			}
		}
		
		if(m != null) {
			outputList.add(m);
		}
		outputList.addAll(tempMainList);
	
		return outputList;
	}
	
	@Override
	public boolean checkCondition(Map<String, AgentData> currentOrgKnowledge) {
		for(Entry<String, AgentData> entry : currentOrgKnowledge.entrySet()){
			
			boolean posRampTriggered = entry.getValue().position > (rampSensor.position + rampSensor.length*MERGEPERCENTAGE);
			boolean rampSensorTriggered = entry.getValue().roadID.equals(rampSensor.lane.getRoadID()) && posRampTriggered;
			boolean posMainTriggered = entry.getValue().position > (mainSensor.position + mainSensor.length*MERGEPERCENTAGE);
			boolean mainSensorTriggered = entry.getValue().roadID.equals(mainSensor.lane.getRoadID()) && posMainTriggered;
			boolean isNotTicked = !tickedAgents.contains(entry.getKey());
			if((rampSensorTriggered || mainSensorTriggered) && isNotTicked)
				return true;
		}
		return false;
	}

	@Override
	public boolean violated(AgentData ad) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge,
			int currentTime) {
		// TODO Auto-generated method stub
		return false;
	}
	
	public static List<Sensor> getSensors(){
		return sensors;
	}


	@Override
	public List<AgentData> getGoals() {
		// TODO Auto-generated method stub
		return null;
	}

}
