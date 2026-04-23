#--------------------------
#----ASSISTANT SFAR---------
if not st.session_state.show_sfar_chat:
    launcher = st.container()
    with launcher:
        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button("Posez votre question clinique", key="open_chat_text", use_container_width=True):
                st.session_state.show_sfar_chat = True
                st.rerun()

        with col2:
            if st.button("💬", key="open_chat_icon", use_container_width=True):
                st.session_state.show_sfar_chat = True
                st.rerun()

    launcher.float(
        float_css_helper(
            bottom="18px",
            right="18px",
            width="300px",
            padding="10px",
            background="white",
            border="1px solid #dcdcdc",
            border_radius="20px",
            box_shadow="0 10px 30px rgba(0,0,0,0.16)",
            z_index="999999"
        )
    )

if st.session_state.show_sfar_chat:
    chat_box = st.container()

    with chat_box:
        top1, top2 = st.columns([8, 1])

        with top1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #c86cff, #8f5cff);
                color: white;
                padding: 18px;
                border-radius: 20px 20px 0 0;
                margin: -16px -16px 10px -16px;
                display:flex;
                justify-content: space-between;
                align-items:center;
            ">
                <div style="font-size:18px; font-weight:700;">
                    Assistant SFAR
                </div>
            </div>
            """, unsafe_allow_html=True)

        with top2:
            if st.button("✕", key="close_chat_simple", use_container_width=True):
                st.session_state.show_sfar_chat = False
                st.rerun()

        messages_box = st.container(height=330)

        with messages_box:
            for msg in st.session_state.messages_sfar:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="sfar-bubble-row"><div class="sfar-bubble-user">{msg["content"]}</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="sfar-bubble-row"><div class="sfar-bubble-assistant">{msg["content"]}</div></div>',
                        unsafe_allow_html=True
                    )

        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

        input_col, send_col = st.columns([8,1])

        with input_col:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            question_sfar = st.text_input(
                "",
                key="question_sfar_float",
                placeholder="Écrivez votre question...",
                label_visibility="collapsed"
            )

        with send_col:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            if st.button("➤", key="send_msg_simple", use_container_width=True):
                if question_sfar.strip():
                    st.session_state.messages_sfar.append({
                        "role": "user",
                        "content": question_sfar
                    })

                    reponse = repondre_assistant_sfar(question_sfar)

                    st.session_state.messages_sfar.append({
                        "role": "assistant",
                        "content": reponse
                    })
                    st.rerun()
