package nl.uu.trafficmas.roadnetwork;

import nl.uu.trafficmas.organisation.BruteState;

public class Lane implements BruteState {
		
	public LaneType laneType;
	public final byte laneIndex;
	private String id;
	private  Lane leftLane;
	private double laneMeanTravelTime;
	private double laneMeanSpeed;
	
	public Lane(LaneType laneType, byte laneIndex) {
		this.laneIndex = laneIndex;
		this.laneType = laneType;
		this.leftLane = null;
		id = null;
		laneMeanTravelTime = Double.MAX_VALUE;
		setLaneMeanSpeed(Double.MAX_VALUE);
	}
	
	public void setLeftLane(Lane leftLane) {
		if(this.leftLane == null)
			this.leftLane = leftLane;
	}
	
	
	public String getID() {
		return id;
	}
	
	public String getRoadID(){
		// TODO: Hackings gezet hier.
		String roadID = this.id.substring(0, id.length()-2);
		return roadID;
	}
	
	public void setID(String id) {
		if(this.id == null)
			this.id = id;
	}
	
	public LaneType getLaneType(){
		return laneType;
	}

	public boolean hasLeftLane() {
		return leftLane != null;
	}
	
	public boolean hasRightLane(){
		return (laneIndex > 0);
	}

	public Lane getLeftLane() {
		return leftLane;
	}

	public double getMeanTravelTime() {
		return laneMeanTravelTime;
	}
	public void setMeanTravelTime(double laneMeanTravelTime) {
		this.laneMeanTravelTime = laneMeanTravelTime;
	}

	public double getLaneMeanSpeed() {
		return laneMeanSpeed;
	}

	public void setLaneMeanSpeed(double laneMeanSpeed) {
		this.laneMeanSpeed = laneMeanSpeed;
	}
}
