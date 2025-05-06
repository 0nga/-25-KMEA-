package nl.uu.trafficmas.tests.norms;

import static org.junit.Assert.*;
import nl.uu.trafficmas.exception.InvalidParameterCombination;
import nl.uu.trafficmas.exception.InvalidDistanceParameter;
import nl.uu.trafficmas.norm.MergeNormScheme;

import org.junit.Test;

public class MergeNormFindBestSpeedTest {


	@Test
	public void noDistRemaining() throws Exception {
		double velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars, result;
		velocity						= 10;
		acceleration					= 1;
		distRemaining					= 0.0;
		lastCarArrivalTimeMergePoint	= 2.0;	
		timeBetweenCars					= 2.0;
		
		MergeNormScheme.findBestSpeed(velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars);
	}
	
	@Test
	public void negDistance() {
		double velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars, result;
		velocity						= 10;
		acceleration					= 1;
		distRemaining					= -100.0;
		lastCarArrivalTimeMergePoint	= 2.0;	
		timeBetweenCars					= 2.0;
		
		MergeNormScheme.findBestSpeed(velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars);
	}

	@Test
	public void negVelocity() {
		double velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars, result;
		velocity						= -22;
		acceleration					= 1;
		distRemaining					= 200.0;
		lastCarArrivalTimeMergePoint	= 10.0;	
		timeBetweenCars					= 2.0;
		
		result = MergeNormScheme.findBestSpeed(velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars);
	}
	
	@Test
	public void normalAccelSituation() {
		double velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars, result;
		velocity						= (80/3.6);
		acceleration					= 1;
		distRemaining					= 300.0;
		lastCarArrivalTimeMergePoint	= 10.0;	
		timeBetweenCars					= 2.0;
		
		result = MergeNormScheme.findBestSpeed(velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars);
		assertEquals(25.42, result, 0.01);
	}
	
	@Test
	public void normalDecelSituation() {
		double velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars, result;
		velocity						= (120/3.6);
		acceleration					= -3;
		distRemaining					= 300.0;
		lastCarArrivalTimeMergePoint	= 10.0;	
		timeBetweenCars					= 2.0;
		
		result = MergeNormScheme.findBestSpeed(velocity, acceleration, distRemaining, lastCarArrivalTimeMergePoint, timeBetweenCars);
		assertEquals(23.71, result, 0.01);
	}
}



