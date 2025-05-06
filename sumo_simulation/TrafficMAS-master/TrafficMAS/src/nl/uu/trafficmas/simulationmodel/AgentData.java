package nl.uu.trafficmas.simulationmodel;

public class AgentData implements Data {

	private static final int LaneSwitchConstant = 2;
	public final String id;
	public final String leaderId;
	public final double leaderDistance;
	public final double position;
	public final double velocity;
	public final String roadID;
	public final int laneIndex;
	public final double deceleration;
	public final double acceleration;

	public AgentData(String id, Object[] leader, double position, double speed, String roadID, int laneIndex, double deceleration, double acceleration) {
		this.id = id;
		if(leader == null) {
			leaderId = null;
			leaderDistance = Double.MIN_VALUE;
			
		} else {
			leaderId = (String) leader[0];
			leaderDistance = (double) leader[1];
		}
		this.position 		= position;
		this.velocity		= speed;
		this.roadID 		= roadID;
		this.laneIndex 		= laneIndex;
		this.deceleration 	= deceleration;
		this.acceleration	= acceleration;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("ID: "+ id+"\r\n");
		sb.append("Speed: "+ velocity+"\r\n");
		sb.append("Position: "+ position+"\r\n");
		sb.append("roadID: "+ roadID+"\r\n");
		sb.append("laneIndex: "+ laneIndex+"\r\n");
		sb.append("LeaderID: "+ leaderId+"\r\n");
		sb.append("LeaderDistance: "+ leaderDistance+"\r\n");

		return sb.toString();
	}

	public static double distance(AgentData agentData, AgentData goal, double acceleration, double deceleration) {
		double laneDist 	= laneDistance(agentData.laneIndex,goal.laneIndex);
		double speedDist	= speedDistance(agentData.velocity, goal.velocity, acceleration, deceleration);
		
		return (laneDist + speedDist);
	}

	private static double speedDistance(double velocityAgent, double velocityGoal,
			double acceleration, double deceleration) {
		//apparently speed is not used in this norm
		if(velocityAgent < 0 || velocityGoal < 0)
			return 0;
		
		boolean shouldDecel = velocityAgent > velocityGoal;
		double accel = (shouldDecel) ? deceleration : acceleration;
		
		return Math.abs((velocityAgent - velocityGoal)/accel);
	}

	private static double laneDistance(int laneIndexAgent, int laneIndexGoal) {
		// TODO Auto-generated method stub
		// apparently laneIndex is not an issue
		if(laneIndexAgent < 0 || laneIndexGoal < 0)
			return 0;
		
		
		return Math.abs(laneIndexAgent-laneIndexGoal)*AgentData.LaneSwitchConstant;
	}
}
