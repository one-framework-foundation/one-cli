# one license, version 0.6.3

**Adopted by Jesse Sep on 2026-10-18 for one-cli releases.**

Jesse Sep adopted this text as-is on 2026-10-18; it has effect for each release that ships it as its license. Any change ships as a new version. Text in "[BLANK: ...]" is a value still to be filled in. Text in "[PROPOSAL: ...]" is a suggested value that stands until someone decides otherwise. Text marked "[SET BY STEWARD DRAFT, researched: ...]" is a value the drafters set under Jesse Sep's delegation after reading the cited Dutch law; each such item still needs counsel to confirm. Nothing in this draft is legal advice.

## Statement of intent

We are determined to find solutions to real-world problems using factual data and well-built arguments that are rooted in reality. We invite any discourse: questions, objections and better arguments are welcome, and they change this text when they are right.

## Plain-language summary (non-binding)

This summary helps you read the license. It is not part of the license and does not change it. Where this summary and the license differ, the license applies.

**Which tier are you?**

- **Tier 0** - use that serves nobody outside your household or organisation. A hobby node at home, a lab using it for its own research. A company's internal use for its commercial work, including research and development for that work, is commercial use and needs a commercial licence (clause 1.6).
- **Tier 1** - small deployment. You serve people outside your household or organisation, and you are below the size threshold in clause 4.10. You confirm that in a short written self-attestation. Lighter records.
- **Tier 2** - everyone else who serves others. Full duties.

| | You can | You must | You cannot |
|---|---|---|---|
| Everyone | Use, study, change and share the software for non-commercial purposes. Combine it with Apache, MIT and similar code. Keep your own separate programs under any license. | Keep the license and notices when you share it. Use secure defaults. Test security only where you are authorized. | Use the name "one" to suggest endorsement. Present a modified version as the original. |
| Tier 0 | Run it for yourself, your family or your own non-commercial organisation. | Nothing beyond "Everyone". | - |
| Tier 1 | Serve others with lighter records. | Keep a self-attestation. Disclose data flows, offer export, give a contact, be honest about AI, ask before training on User data. | Train on User data without separate permission. Lock Users in. |
| Tier 2 | Serve others at any scale. | All Tier 1 duties, plus full per-User records and full records of automated decisions. | As Tier 1. |
| Commercial use (any tier) | Earn from the software, or use it inside a company for commercial work or R&D, once a payout model is in force. | Sign the Commercial Undertaking. Share surplus under the payout model, as one itself and its founders must too. Repay anything taken out in breach. | Use it commercially before then. A side agreement is possible only if it carries the same payout duties. |

Also: every contribution submitted, including teaching, care, design and review, is recorded. When you contribute, you license your contribution under this license and confirm you have the right to. Later versions cannot erase assessed contributions, weaken the core duties, apply backwards, or exempt the Steward or insiders. If you break the terms, you get notice and 30 days to fix it, and fixing a payout breach means paying the money back, at once or in monthly instalments over at most 12 months. Commercial use without a commercial licence is infringement from the start, with no cure period. This is not an OSI open source license (see clause 3.4). How surplus is calculated (the payout model, or "rekensom") is not in this license: it lives in a separate document that is still open. It applies in full to one's own revenue; outside companies are bound by its limits (clause 9.3).

# Part I - Fixed core

The Fixed Core is clauses 1 to 11. It can be changed only within the bounds of clause 11.

## 1. Definitions

1.1 "Work" means the files and revisions that a rights manifest published by the Steward identifies as licensed under this license. The rights manifest records, per file, the authorship evidence the Steward holds.

1.2 "Licensor" means Jesse Sep Sfeerbeheer B.V., a private limited company (besloten vennootschap) registered with the Netherlands Chamber of Commerce (KvK) under number 42068976, and each other rights holder in the Work who grants this license, directly or through the inbound licence in clause 2.5. From the transfer under clause 1.14, the one Foundation takes the place of Jesse Sep Sfeerbeheer B.V.

1.3 "Steward" means the one Foundation once it is incorporated. Until then, Jesse Sep Sfeerbeheer B.V. acts as interim Steward under clauses 1.14 to 1.16. The Steward publishes versions of this license and of the Payout Model and keeps the Contribution Record.

1.4 "You" means the natural or legal person exercising rights under this license, together with its Group.

1.4A "Group" means a person and every entity that controls it, is controlled by it, or is under common control with it. "Control" means direct or indirect ownership of more than half of the voting rights or equity, ownership of substantially all of an entity's assets, or the power to direct its management and policies by vote, contract, or otherwise. Entities acting together under an arrangement whose main purpose is to avoid clause 1.9, and entities acting on Your instructions or substantially funded by You for the use in question, are treated as one Group. A "groep" within the meaning of Book 2 article 24b of the Dutch Civil Code is always a Group, whether or not it meets the control test above. [SET BY STEWARD DRAFT, researched: BW 2:24b defines a group as an economic unit in which legal persons and partnerships are organisationally linked; that test can reach entities the control test misses and vice versa, so the clause takes the union of both rather than replacing one with the other.] Counsel to confirm.

1.5 "Non-Commercial Purpose" means use not primarily intended for or directed towards commercial advantage or monetary compensation. Personal use, and study, research and education not directed towards commercial advantage, are Non-Commercial Purposes. Research or evaluation by or for a for-profit entity, or whose results are used in a commercial product or service, is Commercial Use. Use by a non-profit or public body for its own mission is a Non-Commercial Purpose, except to the extent it provides the Work, or services built on it, to or for a for-profit entity. Charging natural persons or non-profits only the direct cost of a copy or of hosting is not commercial advantage. [SET BY STEWARD DRAFT, researched: Dutch law has no statutory definition of non-commercial use for copyright licences; the wording follows the "commercial advantage or monetary compensation" formula of CC BY-NC 4.0 section 1(i), which is the established reference point, and is kept unchanged.] Counsel to confirm.

1.6 "Commercial Use" means any use of the Work, or of a derivative, that is not for a Non-Commercial Purpose. Internal use by a for-profit organisation for its commercial work, even at Tier 0, and research and development by or for a for-profit organisation for its commercial work, are Commercial Use. Research that meets clause 1.5 remains a Non-Commercial Purpose.

1.7 "Deployment" means a product or service You operate using the Work that processes the data of, or provides AI functionality to, persons outside Your household or, for an organisation, outside its staff acting in that role. Use solely by You, Your household or Your organisation internally ("Tier 0 use") is not a Deployment. Whether Tier 0 use is also Commercial Use is decided by clause 1.6, not by this clause.

1.8 "User" means a natural person whose data a Deployment processes or who interacts with it. "User Data" means a User's inputs, conversations, documents and creations entrusted to a Deployment.

1.9 "Covered Enterprise" means a commercial operation, run by any member of a Group, of a product or service that substantially incorporates the Work or a derivative, or that gives its users access to a substantial set of the Work's features or functionality, as delimited by the attribution method in the Payout Model.
(a) *Wrapper test.* A product or service that mainly passes requests to the Work, or presents the Work's functionality under another interface, name, API or packaging, substantially incorporates the Work. This applies whether the Work is copied, linked, called over a network, or run by another member of the Group or a contractor on its behalf.
(b) *Mere dependency.* Use of the Work is a mere dependency, and does not bring a business into scope, when the Work is not exposed to users and the product's value to its users would not be materially reduced by replacing the Work with a comparable component. Use of an unrelated service is not covered.
(c) *Aggregation.* Every threshold, attribution measure and revenue figure in this license or the Payout Model is computed across the whole Group, together with any revenue that a Group member receives from the operation through licensing, transfer pricing or fees. Splitting an operation across Group members does not reduce its scope.

1.10 "Accountable Operator" means the Group member that controls the Covered Enterprise and the allocation of its attributable value. Every other Group member that receives that value is jointly bound by the Commercial Undertaking for it. Each of them is jointly and severally liable (hoofdelijk verbonden) for the obligations under the Commercial Undertaking for that value. [SET BY STEWARD DRAFT, researched: under BW 6:6(2) debtors are jointly and severally liable only where law, custom or a juristic act provides it, so the clause must say so expressly; under BW 6:7 the creditor may then claim the whole from each. It binds only Group members who sign or accede to the Commercial Undertaking (Annex B.6(c)).] Counsel to confirm.

1.11 "Submission" means any work offered to a project of the Work, accepted or not, including code, documentation, design, review, testing, teaching, translation, care, community and governance work, and infrastructure. "Contribution" means a Submission that has been accepted into a project, or assessed with a value above zero under the Payout Model. A Submission is recorded as proof of receipt only.

1.12 "Payout Model" means the separate document "one rekensom" in the version that applies under clause 11, which sets out how surplus from Commercial Use is calculated and allocated. "Commercial Undertaking" means the signed agreement outlined in Annex B.

1.13 Separate programs, third-party components, model weights, outputs and User Data are not part of the Work merely because they connect to it, are processed by it, or are distributed alongside it. Their own licenses continue to apply. This license grants only rights the Licensor holds, and claims no rights in material that is not protected.

1.14 Chain of title and transfer to the Foundation.

(a) Jesse Sep holds the copyright in the parts of the Work he authored personally. Before the first release under this license: he grants Jesse Sep Sfeerbeheer B.V. a written exclusive licence, with the right to sublicense, limited to licensing the Work under this license and administering clause 9. That licence ends on the transfer under 1.14(b) or (d), when Jesse Sep transfers that copyright by deed directly to the Foundation or the fallback recipient. The deed granting the exclusive licence carries the date on which it is actually signed, and is signed before the first release under this license.

(b) Handover. Jesse Sep Sfeerbeheer B.V. acts as interim Steward under an exemption from clause 9.4 that is limited to clauses 1.14 to 1.16. It shall procure that the one Foundation (stichting) is incorporated by notarial deed no later than 12 months after the first release under this license (the "Handover Deadline"), and shall transfer to it, within 30 days of incorporation, all rights in the Work it holds or controls for this license, its rights and obligations under this license, the Contribution Record, every Commercial Undertaking and the balance of the Ring-Fenced Account. From that transfer, the Foundation is Licensor and Steward in its place, and the exemption ends.

(c) The transfer does not affect licences already granted. Every recipient keeps its licence on the same terms.

(d) Fallback. If the Foundation is not incorporated by the Handover Deadline, or the transfer is not completed within the period above, then: (i) the interim Steward may not accept new Commercial Undertakings or make Commercial Use itself; (ii) it shall, within [PROPOSAL: 60 days], make the same transfer to an existing Dutch stichting, or a fiscal host without shareholders, that has agreed in writing to accept and is named in the rights manifest by the Naming Date under clause 1.17 [BLANK: name of the fallback recipient; candidates in the decisions table, written acceptance required]; and (iii) if that also fails, the Work remains licensed under this license, no person owes payment under clause 9 to the interim Steward, and any balance of the Ring-Fenced Account is distributed under the Payout Model by the independent party named in clause 1.16(c). The Handover Deadline may be extended once, by at most [PROPOSAL: 3 months], by public notice giving the reason.

1.15 Safeguards during the interim. Until the transfer under clause 1.14(b) or (d):

(a) all income the interim Steward receives under this license or a Commercial Undertaking is held in a separate bank account in its name, used only for the purposes of this license (the "Ring-Fenced Account"), and is not available for dividend, shareholder loan, management fee or other distribution to its shareholders, directors or related persons;

(b) costs may be charged to that account only at cost and with receipts; any payment to Jesse Sep or a related person requires prior public notice;

(c) the interim Steward may not pledge, license exclusively, transfer or otherwise encumber the Work, the Contribution Record or the Ring-Fenced Account, except to the Foundation or the fallback recipient under clause 1.14;

(d) it publishes a report at least every [PROPOSAL: 3 months] stating account balance, income, costs, Commercial Undertakings signed and progress toward incorporation;

(e) it adopts a shareholder resolution, and if possible amends its articles, binding itself and its shareholders to clauses 1.14 to 1.16.

1.16 Enforcement of the handover.

(a) Clauses 1.14 to 1.16 are a stipulation for the benefit of every recorded contributor and every Accountable Operator (derdenbeding, Book 6 article 253 of the Dutch Civil Code), who may accept and enforce them, including by a claim for specific performance of the transfer.

(b) A breach of clause 1.15(a) or (c) is a material breach of this license by the interim Steward.

(c) An independent party, a notary or a named trusted person, holds a copy of the deed of transfer in escrow, can execute the fallback, and enforces clause 9 against the interim Steward. It is named in the rights manifest by the Naming Date under clause 1.17. [BLANK: name of the independent party; candidates in the decisions table, written acceptance required]

1.17 Naming Date for the named parties.

(a) The "Naming Date" is the earlier of: (i) the day before the Steward accepts its first Commercial Undertaking, receives any payment under this license or a Commercial Undertaking, or makes Commercial Use itself; and (ii) [PROPOSAL: 6 months] after the first release under this license.

(b) By the Naming Date, the Steward names in the rights manifest the fallback recipient under clause 1.14(d), the independent party under clause 1.16(c) and the counterparty under Annex B, and publishes, or records with the independent party, each one's written acceptance.

(c) Until all three are named and have accepted: (i) the Steward may not accept a Commercial Undertaking, receive payment under this license or make Commercial Use itself; (ii) no Commercial Use licence under clause 2.2 can be granted; and (iii) the non-commercial grants under clause 2.1 are unaffected. A Steward that misses the Naming Date remains bound by (i) and (ii) until it complies.

(d) Clause 1.17 is part of the stipulation in clause 1.16(a).

## 2. Grants

2.0 Effect. No clause of section 2 has effect until a release identifies this license as adopted. "Upon adoption" in this section means from the date of that release.

2.1 Non-commercial grant. Upon adoption by a release, each Licensor grants You a worldwide, non-exclusive, royalty-free licence, for the duration of the relevant rights, to use, copy, modify, distribute, publicly display and perform, and make available the Work and derivatives, for Non-Commercial Purposes only.

2.2 Commercial grant. Upon adoption by a release, each Licensor grants You the same licence for Commercial Use, but only while:

(a) You have signed the Commercial Undertaking and it is in force; and

(b) a version of the Payout Model that the Steward has declared final has been published and has entered into force.

Otherwise You have no licence for Commercial Use under this license. Commercial Use without that licence is outside the scope of the licence granted, not merely a breach of contract. A separate written agreement or an interim license the Steward publishes may permit Commercial Use only if it: (i) includes obligations at least equal to clauses 9 and 10 and the Payout Model then in force or, before one is final, an escrow, held pending it in the Ring-Fenced Account or with the independent party under clause 1.16(c), of all attributable surplus and in no case less than 20% of attributable receipts (ii) is published in summary in the Contribution Record; and (iii) is subject to clause 11.2 as if it were a version.

2.3 Patent grant. Upon adoption by a release, each Licensor grants You, within the scope of clause 2.1 and, where it applies, clause 2.2, a licence under patent claims it can license that are necessarily infringed by its Contribution alone or in combination with the Work as contributed, to make, have made, use, offer for sale, sell, import and otherwise transfer the Work. If You institute or join patent litigation, including a cross-claim or counterclaim, alleging that the Work, a Contribution, or a product to the extent it implements the Work infringes a patent, Your patent licences under this license end on the date it is filed.

2.4 No other rights. No rights are granted by implication or estoppel. A Contribution creates no grant on behalf of anyone who did not authorize it. Statutory exceptions to copyright that apply to You are not reduced.

2.5 Inbound licence. Upon adoption by a release, by offering a Submission to a project of the Work You license it under this license and its later versions, and allow the Steward to sublicense it under clause 2.2. You certify that You have the right to do so, including any consent needed from an employer, by a Developer Certificate of Origin 1.1 sign-off. A Submission without that sign-off is not accepted into a project.

2.6 Acceptance. By using, copying, modifying or distributing the Work You accept this license. If You do not accept it, You have no right to do any act it permits beyond what statute allows. Commercial Use and Deployment additionally require an affirmative act: signing the Commercial Undertaking, or an equivalent recorded acceptance the Steward publishes.

## 3. Distribution and compatibility

3.1 When You distribute the Work or a derivative, include this license and the rights manifest, keep existing copyright, patent, attribution and third-party notices, and mark changed files or provide a change record.

3.2 Every recipient receives a licence directly from each Licensor on these terms. You may not add restrictions on these rights for the covered portions.

3.3 You may license Your own additions and separate programs on any terms, and need not publish their source. You may not present third-party material as licensed under this license.

3.4 Compatibility.

(a) This is a source-available license. It is not an open source license under the Open Source Definition, because it limits commercial use and sets duties on Deployments.

(b) Code under permissive licenses such as Apache 2.0, MIT or BSD may be included in or combined with the Work, provided You keep their notices and, for Apache 2.0, its NOTICE file and patent terms.

(c) The Work cannot be combined with code under the GNU GPL, LGPL or AGPL into a single combined work that is distributed, because those licenses forbid the additional restrictions this license carries. Keep such code in a separate program.

(d) Separate programs that only communicate with the Work, for example over a network interface, command line or file exchange, are not affected by this license, whatever their license. This governs how those programs are licensed; it does not remove Deployment duties under clause 4.0 or scope under clause 1.9.

## 4. Deployment duties

4.0 Scope. These duties apply to each Deployment You control, insofar as it uses the Work, at the tier set by clause 4.10. Where User Data passes through the Work, the duties in clauses 4.1 to 4.7 apply to the Deployment as a whole, including separate programs that provide its user interface or process that data. Clauses 4.6 and 4.8 also apply to Tier 0 use. The duties do not regulate Your unrelated products, Your customers' independent conduct, or other infrastructure. Do not modify or configure the Work to avoid them. They add to applicable law, including data protection, consumer and AI law; they do not replace it. Meeting a duty here does not make processing lawful or a consent valid.

4.1 Local first. Where the Work supports local processing on the User's device or on infrastructure the User controls, make it the default for User Data. Before enabling an external provider, tell the User what will be transferred and why, and obtain an affirmative choice from the User or from an administrator actually authorized to act for that User; technical access alone is not authority. If the User refuses, do not substitute an undisclosed transfer. State which functions need external processing. You need not build a local capability the Work lacks.

4.2 Training consent. Do not use User Data, or let a provider use it, to train or fine-tune models without specific, informed, affirmative permission given separately from ordinary service terms. Ordinary functionality must not depend on it. Explain the data, purposes, recipients and retention. Let the User withdraw permission for future training and request deletion of training copies under Your control. You need not promise to undo completed training. Public availability of data, or holding a copy, is not permission.

4.3 Visible data flows. Before collecting or transferring User Data, including telemetry, disclose categories, purposes, recipients and retention. Obtain separate permission for sharing with third parties beyond what the requested service needs. A materially new purpose or recipient requires renewed disclosure and, where permission was required, renewed permission before it takes effect. On request, give the User an accessible record of external processing of their data, without disclosing secrets or other people's information. Tier 1 meets this last duty under clause 4.11(a).

4.4 Freedom to leave. Provide export of User Data the Deployment retains and controls, in a documented, commonly readable machine format. Do not use technical measures in the Work to prevent switching providers or permitted self-hosting. Disclose actual migration and self-hosting limits. You need not disclose data, credentials or intellectual property You have no authority to disclose.

4.5 Accountable actions. Identify the operator of the Deployment and a contact for questions and corrections. For automated actions that materially affect a person's access, resources or rights, keep appropriate records, explain on request the material inputs, limitations and basis, offer review by a human, and allow further automated actions to be stopped. Trade secrets and others' confidential information need not be disclosed. Tier 1 meets the record duty under clause 4.11(b).

4.6 Secure defaults. Enable by default the authentication and least-privilege access the Work supports, protect User Data in transit and at rest with appropriate encryption the Work supports, and keep credentials out of logs and exports. Document known security limits before affected processing. Provide a private route for vulnerability reports and handle credible reports proportionately. Do not call processing visible to a provider end-to-end encrypted. Reasonable safeguards are required, not perfect security.

4.7 Honest representation. Make clear when a User is interacting with an AI system, and disclose material known limitations. Do not knowingly misrepresent capabilities, uncertainty, actions taken or verification obtained. Do not impersonate a human or use deceptive design to obtain any permission this clause requires. Correct material misleading statements under Your control when You discover them.

4.8 Authorized security testing. When You use the Work to test security, test only an isolated environment You are authorized to control, or obtain the target operator's explicit permission setting targets, methods and limits in advance. Keep testing non-destructive, use synthetic or canary data rather than real secrets, and stop when authorization ends or limits are reached.

4.9 Proportionality. Duties apply in proportion to what the Deployment actually does. A duty tied to a feature the Deployment does not use (for example training, external providers or automated decisions) requires nothing until it uses that feature.

4.10 Tiers.

(a) Tier 0 is use that is not a Deployment (clause 1.7).

(b) Tier 1 is a Deployment operated by You while, counted across You as defined in clause 1.4, all of the following hold: fewer than [PROPOSAL: 100] individuals work for You as employees and independent contractors; Your total revenue in the prior tax year was below [PROPOSAL: EUR 1,000,000, at 2026 prices]; and the Deployment serves fewer than [PROPOSAL: 10,000] Users in a calendar month. The amount is adjusted yearly by [PROPOSAL: the Dutch consumer price index (CPI, all households) published by Statistics Netherlands (CBS), rounded to the nearest EUR 10,000].

(c) Tier 2 is any other Deployment.

(d) To use Tier 1 You keep a written self-attestation that You meet clause 4.10(b), dated and renewed at least once a year, and provide it on request to the Steward or a person giving notice under clause 7. A knowingly false attestation is a material breach.

(e) If You stop meeting clause 4.10(b), You have [PROPOSAL: 90 days] to meet the Tier 2 duties. Splitting an operation across entities, nominees or Deployments to stay below the threshold does not keep it in Tier 1.

These tiers govern Deployment duties only. They do not set any payout threshold; payout thresholds belong to the Payout Model.

4.11 Tier 1 records.

(a) Instead of an individual record under clause 4.3, You may publish and keep current a general description of the external processing the Deployment performs, and answer a User's specific question about their own data within a reasonable time.

(b) Instead of detailed per-action records under clause 4.5, You may keep records sufficient to explain, on request, the type of automated action, its main inputs and the rule or model used, for [PROPOSAL: 12 months after the action]. Human review and the stop control remain required.

(c) All other duties in clause 4 apply to Tier 1 in full.

## 5. Compute for good

5.1 If You distribute an end-user application built on the Work, or operate one as a Tier 2 Deployment, it must offer, or keep where the Work already provides it, a documented optional way for the device owner to contribute idle compute to a humanitarian or open-science project, where the distribution platform permits it. Where it does not, document the option on the project's website.

5.2 The option is off by default, starts only after the resource owner's affirmative choice, offers resource limits and a stop control, and identifies the receiving project, its operator, the resources used and any data transferred. Do not use anyone's machine, money or data without their authority.

5.3 Report contributions accurately on request; zero is a valid report. No minimum, payment, certification or specific provider is required.

5.4 A library or component without an end-user interface need only keep any such functionality and documentation it already contains.

## 6. Marks

No right is granted to use the name "one", logos, trademarks, endorsements or certifications of the Steward or any Licensor, except to describe the origin of the Work accurately. A modified version must not be presented as the unmodified Work.

## 7. Notice and cure

7.0 Scope. Clause 7 applies to breaches of this license and of the Commercial Undertaking. It does not apply to Commercial Use without a licence under clause 2.2, which is infringement from the start.

7.1 Notice. A Licensor or the Steward that believes You are in material breach must give You written notice identifying the conduct, the clause, the date, the cure deadline and a reply route. The notice also serves as notice of default (ingebrekestelling). The cure deadline stated is a reasonable period, and no shorter than the cure period in clause 7.2. [SET BY STEWARD DRAFT, researched: under BW 6:82(1) a notice of default is a written summons to perform within a reasonable period stated in it; BW 6:83 lists when default arises without notice; email counts as written. The added sentence ties the stated period to clause 7.2. Article 6:93 was read only through secondary sources in this job.] Counsel to confirm.

7.1A Addresses and receipt.

(a) Notices to the Steward go to the registered office of the Steward as shown in the Netherlands Chamber of Commerce register (for Jesse Sep Sfeerbeheer B.V., KvK 42068976, the address registered for that number at the time of sending) and to admin@oneframework.to. No street address is written into this license. From the transfer under clause 1.14, the Foundation's registered office and published address apply.

(b) A new address takes effect [PROPOSAL: 14 days] after publication in the rights manifest; the old one stays valid until then. The registered office always remains valid.

(c) Notices to You go to the contact required by clause 4.5, the address in Your Commercial Undertaking, or the registered office of a legal entity. If none is known, publication in the rights manifest together with email to any address You used with the project is sufficient.

(d) Notices to contributors go to the address in their record.

(e) Email is received on the next business day in the Netherlands unless a delivery failure is received; registered post on delivery or first refusal. A notice that does not reach You because of Your own act or omission is treated as received.

(f) Notices may be in English or Dutch.

7.2 Cure. You have 30 days after receipt to: (a) stop the breach; (b) correct its ongoing effects within Your control; (c) where the breach concerns clause 9 or the Payout Model, restore every amount distributed, paid or otherwise transferred in breach of them, with statutory interest (wettelijke rente) from the date of transfer, to the allocation the Payout Model requires, or agree in writing with the Steward a repayment schedule of equal monthly instalments over no more than 12 months, the first falling due within the cure period, and keep to it; and (d) document reasonable mitigation of effects that cannot be undone. Money that has been paid out is never an effect that cannot be undone for the purpose of (d). If You do all of this, Your licence continues.

7.2A Repeat breach. A cure right under clause 7.2 is available once per [PROPOSAL: 12 months] per clause, counted across You as defined in clause 1.4. A repeat breach of the same clause within that period ends Your licence on notice, without a cure period.

7.3 Termination and reinstatement. If You do not cure in time, Your licence under this license ends at the end of that period. A first termination is reinstated automatically once You meet every cure requirement in clause 7.2, including full restoration under clause 7.2(c). Default on an agreed repayment schedule ends the licence again without further notice. After any later termination, reinstatement needs the written agreement of the Licensor concerned or of the Steward. Reinstatement does not waive amounts owed, interest, penalties or damages under the Commercial Undertaking.

7.4 Beneficiaries. Persons entitled to enforce a provision under clause 1.16, clause 9.6, clause 11.4 or the Commercial Undertaking may give notice under clause 7.1 for that provision. A person who gives a manifestly unfounded notice bears the reasonable costs it causes.

7.5 Effects. Patent termination under clause 2.3 is separate. Recipients from You who comply keep their licences. Nothing here permits continued unlawful processing or limits remedies under applicable law. Notice and attribution duties survive for copies lawfully retained.

## 8. Warranty and liability

8.1 To the extent permitted by applicable law, the Work as software is provided "as is", without warranties of any kind, including title, merchantability, fitness for a particular purpose and non-infringement.

8.2 To the extent permitted by applicable law, no Licensor, contributor or the Steward is liable for damage arising from the Work as software.

8.3 Clauses 8.1 and 8.2 do not exclude or limit liability for intent or gross negligence, for death or personal injury caused by negligence, or any liability or right that cannot be excluded by law, including mandatory consumer rights and mandatory product liability. They do not limit duties accepted in a Commercial Undertaking, or duties of the Steward or an Accountable Operator under clauses 1.14 to 1.16, 9, 10 and 11.

8.4 If a provision is invalid or unenforceable, the rest stays in force and the provision applies to the extent it is valid. No restriction is added in its place except by a version published under clause 11.

## 9. Reciprocal non-extraction

9.1 Equal obligation. The Steward, every entity it controls, and every Accountable Operator are bound by this clause 9 and the Payout Model in the same way. The Steward accepts these obligations by signing its own Commercial Undertaking before it makes Commercial Use of the Work.

9.2 No exemption. There is no exemption for founders, investors, operators, the Steward, or its board or members. No exemption can be purchased. This applies to any agreement or interim license under clause 2.2.

9.3 Payout model. How receipts attributable to a Covered Enterprise are counted, which costs and reserves are deducted, and how surplus is allocated is set out in the Payout Model, a separate document. The Payout Model is not final. It is fluid, and this license does not pin it down. Until a version declared final is in force, clause 2.2(b) is not met and no commercial grant exists under this license. Once one is in force, each Accountable Operator accounts and allocates as it requires, and applies its private-benefit bounds before granting ownership, investment, distribution or sale rights that could exceed them. The full Payout Model formula applies to the revenue of the Steward and the entities it controls. Every other Accountable Operator is bound by the bounds in the Payout Model: the outer limit from board Decision 001 (no difference of 100,000 times or more between contributors), no private accumulation beyond the ceiling, transparency of its accounts and allocations, and the structural duties in clauses 9.4, 10 and Annex B.

9.4 Legal form of the Steward. The Steward operates in a legal form without shareholders, such as a foundation or cooperative, and may not transfer the Work, the Contribution Record or its stewardship to a shareholder-owned entity. This applies from the transfer under clause 1.14(b) or (d); until then clauses 1.14 to 1.16 apply. Licensees may have any legal form.

9.5 How this binds. For licensees, clause 9 obligations arise through the Commercial Undertaking, which is the condition of the commercial grant in clause 2.2. They are not imposed on non-commercial users.

9.6 Enforcement.

(a) The Steward enforces clause 9 against every Accountable Operator. The independent party named in clause 1.16(c), and after the transfer under clause 1.14 the party named in the Steward's own Commercial Undertaking, enforces it against the Steward.

(b) Each Commercial Undertaking contains a stipulation for the benefit of the recorded contributors of the Covered Enterprise (derdenbeding, Book 6 article 253 of the Dutch Civil Code). A recorded contributor accepts it by written notice to the Operator, or by bringing a claim. Accepted, it lets the contributor claim restoration under clause 7.2(c) to the allocation the Payout Model requires, and a declaration that a payout was made in breach.

(c) A contributor's claim is for restoration to the allocation, not payment to that contributor personally, except for the contributor's own share.

(d) If the Steward does not act within [PROPOSAL: 60 days] of a written request from a recorded contributor supported by evidence, any recorded contributor may act under (b).

(e) Contributors may act jointly through a foundation or association under Book 3 article 305a of the Dutch Civil Code. Such a body must meet the safeguards of article 305a(2), including a supervisory body, participation mechanisms for the contributors it represents, and sufficient means to bring the claim. [SET BY STEWARD DRAFT, researched: BW 3:305a(1) lets a stichting or vereniging with full legal capacity sue for similar interests of others to the extent its articles cover them and they are sufficiently safeguarded; 3:305a(2) sets representativeness and governance requirements, tested ex nunc in WAMCA cases (Hof Amsterdam 18 June 2024, ECLI:NL:GHAMS:2024:1651). The clause now states those conditions instead of leaving fit open.] Counsel to confirm.

(f) Standing under this clause, clause 7.4 and clause 11.4 belongs to holders of a Contribution relevant to the Operator or text concerned, not to holders of Submissions only.

## 10. Contribution Record

10.1 Duty. The Steward keeps a durable record of every Submission it receives to projects of the Work, including Submissions not accepted. Each Accountable Operator keeps an equivalent record for Contributions relevant to its Covered Enterprise; the Payout Model may allow small operators a simplified record. The record holds only what contributors submit or declare; activity that was not submitted is not tracked.

10.2 Content. A record states the type of Submission, the project, supporting evidence, acceptance status, any assessed value, corrections, the sign-off under clause 2.5, and whether it involved tool assistance, duplicated earlier work or upstream work. A record proves receipt; it does not by itself prove ownership, acceptance or a payment amount. Each contributor has one record. Payouts require verified identity, given to the Steward and not published.

10.3 Facts and identifiers. Contribution facts are kept separately from personal identifiers and sensitive attachments. A contributor may have their record corrected, and personal identifiers erased where the law gives that right. After erasure, non-identifying facts needed to allocate other people's shares are kept, and any claim of the erased person is handled as the Payout Model sets out. Records of payments made, and the evidence on which they were assessed, are kept for the statutory retention period and are pseudonymised rather than deleted to the extent the law permits. The statutory retention period is seven years under Book 2 article 10(3) of the Dutch Civil Code and article 52 of the General Tax Act (Algemene wet inzake rijksbelastingen), or a longer period where another statute requires it. [SET BY STEWARD DRAFT, researched: both provisions set a seven-year retention duty for administration; under GDPR article 6(1)(c) a legal obligation is a lawful basis for keeping the data for that period.] Counsel to confirm.

10.4 Contributor rights. Contributors can submit missing evidence, inspect their record, and contest exclusions and assessments, with reasons published and a route to human appeal. Commit counts, status, investment size or automated assertions alone do not establish value.

10.5 Personal data. The Steward and each Accountable Operator act as controller for the personal data in their record, process only what is necessary, and apply the retention period of [PROPOSAL: the duration of the contributor's record plus 2 years after the contributor's last Submission or payment, whichever is later; records under clause 10.3 that the law requires to be kept longer are kept for that statutory period].

10.6 Rejected Submissions. A Submission that is rejected, spam, or generated without a human contributor's sign-off under clause 2.5 is recorded as received only, gives no standing, and has no assessed value. Rejection is published with a reason and may be contested under clause 10.4.

## 11. Versions

11.1 Later versions. The Steward may publish revised versions of this license and, separately, of the Payout Model, each with a distinct number and a signed hash in the rights manifest. You may use the Work under this version or, at Your option, any later version published by the Steward. A later version takes effect only after the notice period in clause 11.3 expires without a successful objection; no one may rely on a text while an objection under clause 11.4 is pending.

11.2 Bounds. A later version of the license or of the Payout Model must not:

(a) erase, devalue or reassign assessed values of Contributions, or allocations already made;

(b) weaken any duty in clauses 4, 5, 9 or 10, or the bounds in this clause 11, except that the thresholds in clause 4.10(b) may be adjusted by the published index, and a provision held invalid may be replaced by the closest valid provision with the same aim;

(c) add obligations that apply to conduct before its publication;

(d) exempt the Steward, its controlled entities, its board or members, founders, investors or other insiders from any obligation; or

(e) permit transfer of the stewardship to a shareholder-owned entity.

A text that breaches these bounds is not a later version of this license, and the previous version continues to apply. Drafts of the Payout Model that are not declared final are not versions under this clause and bind no one.

11.3 Procedure. The Steward publishes a proposed version, its reasons and changes at least [PROPOSAL: 60 days] before it takes effect, and gives recorded contributors a route to object. [PROPOSAL: Objections are filed in writing with reasons within the notice period and are published with the Steward's answer. The Steward's board decides on the version after the notice period, with written reasons. If recorded contributors holding at least 10% of assessed Contribution value object, the version takes effect only after approval by a majority of the votes cast by recorded contributors, one vote per contributor, in a vote open for at least 14 days.]

11.4 Contributors as beneficiaries. Each recorded contributor may enforce clauses 11.2 and 11.3 against the Steward, including by seeking a declaration that a text is not a valid later version. This is a stipulation for their benefit (derdenbeding, Book 6 article 253 of the Dutch Civil Code), accepted by written notice to the Steward or by bringing a claim.

11.5 No silent change. No website, policy, benchmark, certification scheme, AI output or vote changes this license except through a version published under this clause.

## 12. Governing law and forum

12.1 This license and every Commercial Undertaking are governed by the laws of the Netherlands. Contract disputes go exclusively to the competent court in Amsterdam (rechtbank Amsterdam).

12.2 This does not deprive a consumer of the protection of the mandatory law of their habitual residence, or of the right to sue there.

12.3 Nothing in this clause prevents enforcement of copyright in any competent court under the law of the country for which protection is claimed.

# Annex A - Payout Model (moved out)

The payout model (in Dutch, the "rekensom") is not part of this license. It is kept in the separate document "one rekensom", currently version 0.2, marked NOT FINAL. It is fluid: this license does not pin down its method, shares, thresholds or panel. Clause 2.2(b) and clause 9.3 say what happens until a version is declared final.

# Annex B - Commercial Undertaking (outline)

A signed agreement between the Steward and the Accountable Operator. The Steward signs the same agreement for its own operations, with the independent party under clause 1.16(c), and after the transfer [BLANK: counterparty or independent party; candidates in the decisions table, written acceptance required], as counterparty. Small operators as defined in the Payout Model may sign a short form with the same B.3 to B.6 obligations.

B.1 Parties. Accountable Operator: legal name, registration, address. Steward: Jesse Sep Sfeerbeheer B.V. (KvK 42068976) as interim Steward, succeeded by the one Foundation under clause 1.14(b).

B.2 Scope. The Covered Enterprise and the attribution method of the Payout Model used for it.

B.3 Undertakings. The Operator accepts clauses 9 and 10 and the Payout Model in force, keeps the records they require, files the reports the Payout Model requires, and applies the private-benefit bounds before granting ownership, investment or sale rights.

B.4 Beneficiaries. Recorded contributors relevant to the Covered Enterprise may enforce B.3, B.5 and B.6 directly, under the stipulation in clause 9.6(b).

B.5 Transfer and insolvency. On a sale, merger or transfer of the Covered Enterprise, the Operator binds the acquirer to this undertaking. Records and allocations already made survive the Operator's exit from commercial use.

B.6 Review and remedies.

(a) Review. An independent reviewer appointed by the Steward reviews the Operator's records at the triggers in the Payout Model, on a credible written complaint by a recorded contributor, or once every [PROPOSAL: 2 years]. Reviewed records are confidential and used only for this undertaking. The Operator bears review costs if the review finds a shortfall of more than [PROPOSAL: 5%] of amounts to be allocated; otherwise the Steward bears them.

(b) Restoration. Any amount distributed, paid or transferred in breach of clause 9 or the Payout Model must be restored under clause 7.2(c), with statutory interest. Restoration is a primary obligation of this undertaking, not damages, and is owed whether or not the licence is later reinstated.

(c) Recipient liability. The Operator procures that any director, shareholder, founder or affiliate who received such an amount repays it; the Operator remains liable for amounts not repaid. Recipients who are affiliates sign accession to this clause before receiving distributions. This applies to the interim Steward and its shareholders in the same way (clause 9.2).

(d) Late-restoration penalty. If the Operator has not restored within the cure period, or misses an instalment under clause 7.2(c), it owes a penalty for the delay of 10% of the amount then unrestored, plus 1% for each further full month of delay, capped at 50% of that amount. The penalty is stipulated for delay only. It never replaces restoration: restoration under (b) is always owed in full, and to the extent the penalty cannot be claimed beside restoration, the penalty lapses and restoration stands. The penalty is paid into the allocation pool under the Payout Model, not to the Steward. The court's power to reduce it under Book 6 article 94 of the Dutch Civil Code remains. [SET BY STEWARD DRAFT, researched: under BW 6:92 a creditor cannot claim both the penalty and performance of the obligation unless the penalty is stipulated for mere delay, and a penalty replaces statutory damages; case law accepts cumulation of a delay penalty with performance (for example Hof Den Haag 31 March 2020, ECLI:NL:GHDHA:2020:992, with reduction under 6:94). The clause is already drafted as a delay-only penalty with a lapse rule, so it is kept unchanged.] Counsel to confirm.

(e) Other remedies. Notice and cure under clause 7 apply. Termination under clause 7.3, injunction, damages and claims under Book 6 articles 203 and 212 of the Dutch Civil Code remain available. None of these is exclusive.

(f) Proportionality. Remedies are applied proportionately to the size of the Covered Enterprise and the seriousness and intent of the breach. Small operators as defined in the Payout Model receive a [PROPOSAL: 60-day] cure period.

B.7 Term. In force while the Operator makes Commercial Use, and afterwards for obligations already arisen.

B.8 Law and forum. As in clause 12, unless otherwise agreed in writing.
