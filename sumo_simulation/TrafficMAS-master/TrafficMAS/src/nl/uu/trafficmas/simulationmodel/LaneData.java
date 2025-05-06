package nl.uu.trafficmas.simulationmodel;

public class LaneData implements Data{

	public final String id;
	public final double meanSpeed;
	public final double meanTime;
	public final String edgeID;
	public final double length;

	public LaneData(String id, double meanSpeed, double meanTime, String edgeID, double length) {
		this.id = id;
		this.meanSpeed = meanSpeed;
		this.meanTime = meanTime;
		this.edgeID = edgeID;
		this.length = length;
	}

}
