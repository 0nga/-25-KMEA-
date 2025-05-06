package nl.uu.trafficmas.agent.actions;

public class ChangeVelocityMinus1Action extends ChangeVelocityAction {
	public ChangeVelocityMinus1Action(int priority) {
		super(priority);
		this.speedIncrease 	= -1;
	}
}
