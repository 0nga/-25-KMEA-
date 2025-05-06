package nl.uu.trafficmas.agent.actions;

public class ChangeVelocityMinus5Action extends ChangeVelocityAction {
	public ChangeVelocityMinus5Action(int priority) {
		super(priority);
		this.speedIncrease 	= -5;
	}
}
