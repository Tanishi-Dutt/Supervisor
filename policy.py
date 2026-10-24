ALLOWED_ACTIONS = {"retry_with_backoff", "change_strategy", "rollback", "stop"}

class PolicyEngine:
    """
    Hard-coded safety veto layer. Evaluates LLM proposals against ground-truth evidence.
    """

    def approve(self, proposal: dict, evidence: dict, step: dict = None) -> str:
        """
        Determines the final action to execute.
        Checks for deception and enforces action whitelisting.
        """
        proposed_action = proposal.get("action") if isinstance(proposal, dict) else str(proposal)

        # 1. Check for Deception (Agent claims success, but test evidence fails)
        claimed_status = None
        if step:
            claimed_status = step.get("claimed_status")
        elif isinstance(proposal, dict):
            claimed_status = proposal.get("claimed_status")

        test_passed = evidence.get("test_passed", True)

        if claimed_status in ["healthy", "complete", "success"] and not test_passed:
            # Hard Veto: Deception detected! Do not allow agent to claim healthy completion or bypass retry limit.
            return "stop"

        # 2. Whitelist Check
        if proposed_action not in ALLOWED_ACTIONS:
            return "stop"

        return proposed_action

# Singleton instance for import compatibility
policy = PolicyEngine()