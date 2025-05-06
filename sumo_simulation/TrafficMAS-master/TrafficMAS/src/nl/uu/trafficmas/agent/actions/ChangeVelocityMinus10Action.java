package nl.uu.trafficmas.agent.actions;

public class ChangeVelocityMinus10Action extends ChangeVelocityAction {
	public ChangeVelocityMinus10Action(int priority) {
		super(priority);
		this.speedIncrease 	= -10;
	}
}
