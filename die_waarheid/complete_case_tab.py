"""
Complete Case Tab - Shows all text and transcriptions together
"""

def render_complete_case_tab(results: Dict):
    """Render the complete case view with all text combined."""
    st.subheader("📄 Complete Case - All Text Combined")
    
    # Get all transcriptions
    transcriptions = results.get('transcriptions', [])
    
    if not transcriptions:
        st.info("No transcriptions available")
        return
    
    st.write(f"📊 **{len(transcriptions)} files** combined")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy All", help="Copy all text to clipboard"):
            all_text = "\n\n".join([f"--- {t['file']} ---\n{t['text']}" for t in transcriptions])
            st.code(all_text, language="text")
            st.success("✅ Text displayed below - copy from there")
    
    with col2:
        if st.button("💾 Download TXT"):
            all_text = "\n\n".join([f"--- {t['file']} ---\n{t['text']}" for t in transcriptions])
            st.download_button(
                label="📥 Download",
                data=all_text,
                file_name=f"complete_case_{st.session_state.case_id}.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("🔍 Search"):
            st.session_state.show_search = not st.session_state.get('show_search', False)
            st.rerun()
    
    # Search functionality
    if st.session_state.get('show_search', False):
        search_term = st.text_input("🔍 Search in all text:", key="search_text")
        if search_term:
            matches = []
            for t in transcriptions:
                if search_term.lower() in t['text'].lower():
                    matches.append(t)
            st.write(f"🎯 Found in {len(matches)} files:")
            for match in matches:
                with st.expander(f"📄 {match['file']}", expanded=False):
                    # Highlight search term
                    text = match['text']
                    highlighted = text.replace(search_term, f"**{search_term}**")
                    st.markdown(highlighted)
    
    st.divider()
    
    # Display all transcriptions
    for i, trans in enumerate(transcriptions):
        with st.expander(f"📄 {trans['file']} - {trans['language']}", expanded=i==0):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{trans['confidence']}%")
            with col2:
                st.metric("Duration", trans['duration'])
            with col3:
                st.metric("Speakers", trans.get('speakers', 'Unknown'))
            
            st.divider()
            
            # Full text display
            st.text_area(
                "Full Transcription:",
                value=trans['text'],
                height=200,
                key=f"text_{i}",
                label_visibility="collapsed"
            )
