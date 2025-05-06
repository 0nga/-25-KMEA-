package nl.uu.trafficmas.roadnetwork;

import java.util.List;

import nl.uu.trafficmas.simulationmodel.AgentData;
import nl.uu.trafficmas.simulationmodel.SensorData;

public class Sensor {

	public final String id;
	public final Lane lane;
	public final double length;
	public final double position;
	public final int frequency;
	private SensorData sensorData;
	
	public Sensor(String id, Lane lane, double position, double length, int frequency){
		this.id 		= id;
		this.lane		= lane;
		this.position	= position;
		this.length		= length;
		this.frequency	= frequency;
	}
	
	public String getId(){
		return id;
	}
	
	public Lane getLane(){
		return lane;
	}
	
	public double getPosition(){
		return position;
	}
	
	public double getLength(){
		return length;
	}
	
	public double getEndPosition() {
		return this.position+this.length;
	}

	public List<AgentData> readSensor() {
		// TODO Auto-generated method stub
		return sensorData.getAgentsData();
	}

	public void updateSensorData(SensorData sensorData) {
		// TODO Auto-generated method stub
		this.sensorData = sensorData;
		
	}
	public int getFrequency(){
		return frequency;
	}

	public double getLastStepMeanSpeed() {
		// TODO Auto-generated method stub
		return this.sensorData.meanSpeed;
	}
}
